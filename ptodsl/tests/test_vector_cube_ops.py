# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

import ptodsl._ops as _ops
from ptodsl import pto


def _identity(value):
    return value


class VectorCubeSurfaceTest(unittest.TestCase):
    def test_public_namespace_exports_new_vector_and_cube_apis(self):
        names = [
            "vsub", "vmin", "vand", "vor", "vxor", "vshl", "vshr",
            "vln", "vsqrt", "vabs", "vneg", "vrec", "vrsqrt", "vrelu", "vnot",
            "vcmin", "vcgmin", "vcpadd",
            "vadds", "vmuls", "vmaxs", "vmins", "vlrelu",
            "vaxpy", "vaddrelu", "vsubrelu", "vsel",
            "mte_gm_l1", "mte_l1_ub", "mte_gm_l1_frac", "mte_l1_bt", "mte_l1_fb",
            "mad_acc", "mad_bias", "mad_mx", "mad_mx_acc", "mad_mx_bias",
            "FractalMode", "AccStoreUnitFlagCtrl", "MadUnitFlagMode", "SatMode", "Tf32Mode", "SplitMode",
        ]

        for name in names:
            self.assertTrue(hasattr(pto, name), name)

    def test_tile_bitwise_aliases_are_exposed_without_legacy_names(self):
        preferred_names = [
            "bit_not", "bit_and", "bit_ands", "bit_or", "bit_ors",
            "bit_xor", "bit_xors", "bit_shl", "bit_shls", "bit_shr", "bit_shrs",
        ]
        legacy_names = [
            "not_", "and_", "ands", "or_", "ors", "xor", "xors", "shl", "shls", "shr", "shrs",
        ]

        for name in preferred_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pto.tile, name), name)

        for name in legacy_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(pto.tile, name), name)

    def test_tile_partial_and_fillpad_names_are_exposed_without_legacy_names(self):
        preferred_names = [
            "partadd", "partmul", "partmax", "partmin",
            "fillpad", "fillpad_expand", "fillpad_inplace",
        ]
        legacy_names = [
            "part_add", "part_mul", "part_max", "part_min",
            "fill_pad", "fill_pad_expand", "fill_pad_inplace",
        ]

        for name in preferred_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pto.tile, name), name)

        for name in legacy_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(pto.tile, name), name)

    def test_sync_flag_names_are_exposed_without_legacy_aliases(self):
        preferred_names = [
            "set_cross_flag", "wait_cross_flag",
            "set_intra_flag", "wait_intra_flag",
        ]
        legacy_names = [
            "set_cross_core", "wait_flag_dev",
            "set_intra_block", "wait_intra_core",
        ]

        for name in preferred_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pto, name), name)

        for name in legacy_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(pto, name), name)

    def test_direct_vector_wrappers_dispatch_to_generated_ops(self):
        lhs = SimpleNamespace(type="vec_ty")
        rhs = SimpleNamespace(type="vec_ty")
        mask = SimpleNamespace(type="mask_ty")
        result = object()

        binary_cases = [
            ("vsub", "VsubOp", (lhs, rhs, mask)),
            ("vmin", "VminOp", (lhs, rhs, mask)),
            ("vand", "VandOp", (lhs, rhs, mask)),
            ("vor", "VorOp", (lhs, rhs, mask)),
            ("vxor", "VxorOp", (lhs, rhs, mask)),
            ("vshl", "VshlOp", (lhs, rhs, mask)),
            ("vshr", "VshrOp", (lhs, rhs, mask)),
        ]
        unary_cases = [
            ("vln", "VlnOp", (lhs, mask)),
            ("vsqrt", "VsqrtOp", (lhs, mask)),
            ("vabs", "VabsOp", (lhs, mask)),
            ("vneg", "VnegOp", (lhs, mask)),
            ("vrelu", "VreluOp", (lhs, mask)),
            ("vnot", "VnotOp", (lhs, mask)),
            ("vcmin", "VcminOp", (lhs, mask)),
            ("vcpadd", "VcpaddOp", (lhs, mask)),
        ]

        with patch.object(_ops, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops, "wrap_surface_value", side_effect=_identity):
            for func_name, op_name, args in binary_cases + unary_cases:
                with self.subTest(func=func_name):
                    fake_op = SimpleNamespace(result=result)
                    with patch.object(_ops._pto, op_name, return_value=fake_op) as op_ctor:
                        output = getattr(_ops, func_name)(*args)
                    self.assertIs(output, result)
                    self.assertEqual(op_ctor.call_args.args[0], "vec_ty")

    def test_vec_scalar_wrappers_and_vaxpy_coerce_scalar_operands(self):
        vec = SimpleNamespace(type="vec_ty")
        other = SimpleNamespace(type="vec_ty")
        mask = SimpleNamespace(type="mask_ty")
        scalar = object()
        coerced_scalar = object()
        result = object()

        vec_scalar_cases = [
            ("vadds", "VaddsOp"),
            ("vmuls", "VmulsOp"),
            ("vmaxs", "VmaxsOp"),
            ("vmins", "VminsOp"),
            ("vlrelu", "VlreluOp"),
        ]

        with patch.object(_ops, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops, "wrap_surface_value", side_effect=_identity), \
             patch.object(_ops, "_coerce_scalar_like_vector_element", return_value=coerced_scalar) as coerce_scalar:
            for func_name, op_name in vec_scalar_cases:
                with self.subTest(func=func_name):
                    fake_op = SimpleNamespace(result=result)
                    with patch.object(_ops._pto, op_name, return_value=fake_op) as op_ctor:
                        output = getattr(_ops, func_name)(vec, scalar, mask)
                    self.assertIs(output, result)
                    self.assertEqual(op_ctor.call_args.args, ("vec_ty", vec, coerced_scalar, mask))

            fake_op = SimpleNamespace(result=result)
            with patch.object(_ops._pto, "VaxpyOp", return_value=fake_op) as op_ctor:
                output = _ops.vaxpy(scalar, vec, other, mask)
            self.assertIs(output, result)
            self.assertEqual(op_ctor.call_args.args, ("vec_ty", vec, other, coerced_scalar, mask))
            self.assertGreaterEqual(coerce_scalar.call_count, len(vec_scalar_cases) + 1)

    def test_composed_vector_wrappers_chain_existing_primitives(self):
        vec = object()
        rhs = object()
        mask = object()
        zero_vec = object()
        one_vec = object()
        sqrt_vec = object()
        add_vec = object()
        sub_vec = object()
        relu_vec = object()
        reciprocal_vec = object()

        with patch.object(_ops, "vmuls", return_value=zero_vec) as vmuls, \
             patch.object(_ops, "vadds", return_value=one_vec) as vadds, \
             patch.object(_ops, "vdiv", return_value=reciprocal_vec) as vdiv:
            self.assertIs(_ops.vrec(vec, mask), reciprocal_vec)
            vmuls.assert_called_once_with(vec, 0, mask)
            vadds.assert_called_once_with(zero_vec, 1, mask)
            vdiv.assert_called_once_with(one_vec, vec, mask)

        with patch.object(_ops, "vsqrt", return_value=sqrt_vec) as vsqrt, \
             patch.object(_ops, "vrec", return_value=reciprocal_vec) as vrec:
            self.assertIs(_ops.vrsqrt(vec, mask), reciprocal_vec)
            vsqrt.assert_called_once_with(vec, mask)
            vrec.assert_called_once_with(sqrt_vec, mask)

        with patch.object(_ops, "vadd", return_value=add_vec) as vadd, \
             patch.object(_ops, "vrelu", return_value=relu_vec) as vrelu:
            self.assertIs(_ops.vaddrelu(vec, rhs, mask), relu_vec)
            vadd.assert_called_once_with(vec, rhs, mask)
            vrelu.assert_called_once_with(add_vec, mask)

        with patch.object(_ops, "vsub", return_value=sub_vec) as vsub, \
             patch.object(_ops, "vrelu", return_value=relu_vec) as vrelu:
            self.assertIs(_ops.vsubrelu(vec, rhs, mask), relu_vec)
            vsub.assert_called_once_with(vec, rhs, mask)
            vrelu.assert_called_once_with(sub_vec, mask)

    def test_vcgmin_and_vsel_dispatch_correctly(self):
        vec = SimpleNamespace(type="vec_ty")
        other = SimpleNamespace(type="vec_ty")
        mask = SimpleNamespace(type="mask_ty")
        reduced = object()
        scalar = object()
        selected = object()

        with patch.object(_ops, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops, "wrap_surface_value", side_effect=_identity), \
             patch.object(_ops, "_extract_lowest_lane_scalar", return_value=scalar) as extract_scalar, \
             patch.object(_ops._pto, "VcgminOp", return_value=SimpleNamespace(result=reduced)) as vcgmin_op:
            output = _ops.vcgmin(vec, mask)
        self.assertIs(output, scalar)
        self.assertEqual(vcgmin_op.call_args.args, ("vec_ty", vec, mask))
        extract_scalar.assert_called_once_with(reduced, mask)

        with patch.object(_ops, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops, "wrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "VselOp", return_value=SimpleNamespace(result=selected)) as vsel_op:
            output = _ops.vsel(vec, other, mask)
        self.assertIs(output, selected)
        self.assertEqual(vsel_op.call_args.args, ("vec_ty", vec, other, mask))

    def test_cube_variant_wrappers_dispatch_to_generated_ops(self):
        lhs = object()
        rhs = object()
        dst = object()
        bias = object()

        cube_cases = [
            ("mad_acc", "MadAccOp", (lhs, rhs, dst, 1, 2, 3), (lhs, rhs, dst, "i64:1", "i64:2", "i64:3")),
            ("mad_bias", "MadBiasOp", (lhs, rhs, dst, bias, 1, 2, 3), (lhs, rhs, dst, bias, "i64:1", "i64:2", "i64:3")),
            ("mad_mx", "MadMxOp", (lhs, rhs, dst, 1, 2, 3), (lhs, rhs, dst, "i64:1", "i64:2", "i64:3")),
            ("mad_mx_acc", "MadMxAccOp", (lhs, rhs, dst, 1, 2, 3), (lhs, rhs, dst, "i64:1", "i64:2", "i64:3")),
            ("mad_mx_bias", "MadMxBiasOp", (lhs, rhs, dst, bias, 1, 2, 3), (lhs, rhs, dst, bias, "i64:1", "i64:2", "i64:3")),
        ]

        with patch.object(_ops, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops, "_coerce_i64", side_effect=lambda value, *, context: f"i64:{value}"):
            for func_name, op_name, args, expected_call in cube_cases:
                with self.subTest(func=func_name):
                    op_ctor = MagicMock()
                    with patch.object(_ops._pto, op_name, op_ctor):
                        getattr(_ops, func_name)(*args)
                    self.assertEqual(op_ctor.call_args.args, expected_call)

    def test_mad_option_wrappers_dispatch_to_generated_ops(self):
        lhs = object()
        rhs = object()
        dst = object()
        bias = object()
        mad_options = {
            "unit_flag_mode": "unit_flag_attr",
            "disable_gemv": True,
            "sat_mode": "sat_attr",
            "tf32_mode": "tf32_attr",
            "n_dir": True,
        }
        mad_mx_options = {
            "unit_flag_mode": "unit_flag_attr",
            "disable_gemv": True,
            "sat_mode": "sat_attr",
            "n_dir": True,
        }

        with patch.object(_ops, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops, "_coerce_i64", side_effect=lambda value, *, context: f"i64:{value}"), \
             patch.object(_ops, "_mad_options", return_value=mad_options) as normalize_mad, \
             patch.object(_ops._pto, "MadOp", MagicMock()) as mad_op:
            _ops.mad(
                lhs,
                rhs,
                dst,
                1,
                2,
                3,
                unit_flag=pto.MadUnitFlagMode.CHECK_ONLY,
                disable_gemv=True,
                sat=pto.SatMode.OFF,
                tf32_mode=pto.Tf32Mode.ROUND_EVEN,
                n_dir=True,
            )
        normalize_mad.assert_called_once_with(
            unit_flag=pto.MadUnitFlagMode.CHECK_ONLY,
            disable_gemv=True,
            sat=pto.SatMode.OFF,
            tf32_mode=pto.Tf32Mode.ROUND_EVEN,
            n_dir=True,
        )
        self.assertEqual(mad_op.call_args.args, (lhs, rhs, dst, "i64:1", "i64:2", "i64:3"))
        self.assertEqual(mad_op.call_args.kwargs, mad_options)

        with patch.object(_ops, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops, "_coerce_i64", side_effect=lambda value, *, context: f"i64:{value}"), \
             patch.object(_ops, "_mad_mx_options", return_value=mad_mx_options) as normalize_mx, \
             patch.object(_ops._pto, "MadMxBiasOp", MagicMock()) as mad_mx_bias_op:
            _ops.mad_mx_bias(
                lhs,
                rhs,
                dst,
                bias,
                1,
                2,
                3,
                unit_flag=pto.MadUnitFlagMode.CHECK_AND_SET,
                disable_gemv=True,
                sat=pto.SatMode.ON,
                n_dir=True,
            )
        normalize_mx.assert_called_once_with(
            unit_flag=pto.MadUnitFlagMode.CHECK_AND_SET,
            disable_gemv=True,
            sat=pto.SatMode.ON,
            n_dir=True,
        )
        self.assertEqual(mad_mx_bias_op.call_args.args, (lhs, rhs, dst, bias, "i64:1", "i64:2", "i64:3"))
        self.assertEqual(mad_mx_bias_op.call_args.kwargs, mad_mx_options)

    def test_mte_l0c_ub_dst_mode_accepts_enum_like_subblock_value(self):
        enum_like = SimpleNamespace(value=1)
        with patch.object(_ops, "_acc_store_ub_dst_mode_attr", return_value="single_attr"), \
             patch.object(_ops, "_coerce_i64", side_effect=lambda value, *, context: f"i64:{value}"):
            attr, sub_blockid = _ops._mte_l0c_ub_dst_mode(enum_like)
        self.assertEqual(attr, "single_attr")
        self.assertEqual(sub_blockid, "i64:1")

    def test_mte_l0c_ub_dst_mode_accepts_split_enum(self):
        with patch.object(_ops, "_acc_store_ub_dst_mode_attr", side_effect=lambda mode: f"{mode}_attr"):
            attr, sub_blockid = _ops._mte_l0c_ub_dst_mode(split=pto.SplitMode.N)
        self.assertEqual(attr, "split_n_attr")
        self.assertIsNone(sub_blockid)

    def test_cube_sat_modes_map_to_backend_tokens(self):
        with patch.object(_ops, "Attribute") as attr:
            attr.parse.side_effect = lambda text: text
            self.assertEqual(_ops._mad_sat_attr(pto.SatMode.ON), "#pto<mad_sat_mode sat>")
            self.assertEqual(_ops._mad_sat_attr(pto.SatMode.OFF), "#pto<mad_sat_mode nosat>")
            self.assertEqual(_ops._acc_store_sat_attr(pto.SatMode.PRESERVE_NAN), "#pto<acc_store_sat_mode sat_preserve_nan>")

    def test_tile_selection_surface_exposes_optional_tmp(self):
        for func, expected in [
            (_ops.tsel, ["mask", "src0", "src1", "dst", "tmp"]),
            (_ops.tsels, ["mask", "src", "scalar", "dst", "tmp"]),
            (pto.tile.sel, ["mask", "src0", "src1", "dst", "tmp"]),
            (pto.tile.sels, ["mask", "src", "scalar", "dst", "tmp"]),
        ]:
            with self.subTest(func=func):
                signature = inspect.signature(func)
                self.assertEqual(list(signature.parameters.keys()), expected)
                self.assertEqual(signature.parameters["tmp"].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIsNone(signature.parameters["tmp"].default)

    def test_tile_selection_wrappers_use_explicit_tmp_or_synthesize_one(self):
        mask = object()
        src0 = object()
        src1 = object()
        src = object()
        dst = object()
        tmp = object()
        scalar = object()
        coerced_scalar = object()
        synthesized_tmp = object()

        with patch.object(_ops, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops, "_coerce_tile_scalar_operand", return_value=coerced_scalar):
            with patch.object(_ops, "_resolve_selection_tmp", return_value=synthesized_tmp) as resolve_tmp, \
                 patch.object(_ops._pto, "tsel") as tsel_op:
                _ops.tsel(mask, src0, src1, dst)
            resolve_tmp.assert_called_once_with(dst, None, context="tsel")
            self.assertEqual(tsel_op.call_args.args, (mask, src0, src1, synthesized_tmp, dst))

            with patch.object(_ops, "_resolve_selection_tmp", side_effect=AssertionError("should not synthesize")), \
                 patch.object(_ops._pto, "tsel") as tsel_op:
                _ops.tsel(mask, src0, src1, dst, tmp=tmp)
            self.assertEqual(tsel_op.call_args.args, (mask, src0, src1, tmp, dst))

            with patch.object(_ops, "_resolve_selection_tmp", return_value=synthesized_tmp) as resolve_tmp, \
                 patch.object(_ops._pto, "tsels") as tsels_op:
                _ops.tsels(mask, src, scalar, dst)
            resolve_tmp.assert_called_once_with(dst, None, context="tsels")
            self.assertEqual(tsels_op.call_args.args, (mask, src, synthesized_tmp, coerced_scalar, dst))

            with patch.object(_ops, "_resolve_selection_tmp", side_effect=AssertionError("should not synthesize")), \
                 patch.object(_ops._pto, "tsels") as tsels_op:
                _ops.tsels(mask, src, scalar, dst, tmp=tmp)
            self.assertEqual(tsels_op.call_args.args, (mask, src, tmp, coerced_scalar, dst))

    def test_tile_row_reductions_expose_optional_tmp_and_synthesize_one(self):
        src = SimpleNamespace(type="src_ty")
        dst = object()
        tmp = object()
        synthesized_tmp = object()

        row_cases = [
            ("rowsum", "trowsum"),
            ("rowmax", "trowmax"),
            ("rowmin", "trowmin"),
            ("rowprod", "trowprod"),
            ("rowargmax", "trowargmax"),
            ("rowargmin", "trowargmin"),
        ]

        for name, low_level_name in row_cases:
            with self.subTest(func=name):
                signature = inspect.signature(getattr(pto.tile, name))
                self.assertEqual(list(signature.parameters.keys()), ["src", "dst", "tmp"])
                self.assertEqual(signature.parameters["tmp"].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIsNone(signature.parameters["tmp"].default)

                with patch.object(_ops, "unwrap_surface_value", side_effect=_identity), \
                     patch.object(_ops, "alloc_tile", return_value=synthesized_tmp) as alloc_tile, \
                     patch.object(_ops, low_level_name) as low_level_op:
                    getattr(pto.tile, name)(src, dst)
                alloc_tile.assert_called_once_with(tile_type="src_ty")
                low_level_op.assert_called_once_with(src, synthesized_tmp, dst)

                with patch.object(_ops, "unwrap_surface_value", side_effect=_identity), \
                     patch.object(_ops, "alloc_tile", side_effect=AssertionError("should not synthesize")), \
                     patch.object(_ops, low_level_name) as low_level_op:
                    getattr(pto.tile, name)(src, dst, tmp=tmp)
                low_level_op.assert_called_once_with(src, tmp, dst)

    def test_sync_event_id_rejects_out_of_range_static_values(self):
        cases = [
            (_ops.set_flag, ("MTE2", "V"), {"event_id": 8}, "set_flag(..., event_id=...)"),
            (_ops.wait_flag, ("MTE2", "V"), {"event_id": -1}, "wait_flag(..., event_id=...)"),
            (_ops.set_cross_flag, (pto.Pipe.FIX, 8), {}, "set_cross_flag(..., event_id=...)"),
            (_ops.wait_cross_flag, (pto.Pipe.FIX, -1), {}, "wait_cross_flag(..., event_id=...)"),
            (_ops.set_intra_flag, (pto.Pipe.MTE3, 9), {}, "set_intra_flag(..., event_id=...)"),
            (_ops.wait_intra_flag, (pto.Pipe.V, -2), {}, "wait_intra_flag(..., event_id=...)"),
        ]

        with patch.object(_ops._pto, "set_flag") as set_flag_op, \
             patch.object(_ops._pto, "set_flag_dyn") as set_flag_dyn_op, \
             patch.object(_ops._pto, "wait_flag") as wait_flag_op, \
             patch.object(_ops._pto, "wait_flag_dyn") as wait_flag_dyn_op, \
             patch.object(_ops._pto, "sync_set") as sync_set_op, \
             patch.object(_ops._pto, "sync_wait") as sync_wait_op:
            for func, args, kwargs, context in cases:
                with self.subTest(func=func.__name__, event_id=kwargs.get("event_id", args[-1])):
                    with self.assertRaises(ValueError) as exc:
                        func(*args, **kwargs)
                    message = str(exc.exception)
                    self.assertIn(context, message)
                    self.assertIn("[0, 7]", message)

        set_flag_op.assert_not_called()
        set_flag_dyn_op.assert_not_called()
        wait_flag_op.assert_not_called()
        wait_flag_dyn_op.assert_not_called()
        sync_set_op.assert_not_called()
        sync_wait_op.assert_not_called()

    def test_sync_facades_reject_illegal_pipe_endpoints(self):
        cases = [
            (_ops.set_cross_flag, (pto.Pipe.V, 0), "set_cross_flag(pipe, event_id)", "<PIPE_FIX>", "<PIPE_V>"),
            (_ops.wait_cross_flag, (pto.Pipe.MTE3, 0), "wait_cross_flag(pipe, event_id)", "<PIPE_FIX>", "<PIPE_MTE3>"),
            (_ops.set_intra_flag, (pto.Pipe.FIX, 0), "set_intra_flag(pipe, event_id)", "<PIPE_MTE3>", "<PIPE_FIX>"),
            (_ops.wait_intra_flag, (pto.Pipe.MTE3, 0), "wait_intra_flag(pipe, event_id)", "<PIPE_V>", "<PIPE_MTE3>"),
        ]

        with patch.object(_ops._pto, "sync_set") as sync_set_op, \
             patch.object(_ops._pto, "sync_wait") as sync_wait_op:
            for func, args, context, expected, actual in cases:
                with self.subTest(func=func.__name__, pipe=args[0]):
                    with self.assertRaises(ValueError) as exc:
                        func(*args)
                    message = str(exc.exception)
                    self.assertIn(context, message)
                    self.assertIn(expected, message)
                    self.assertIn(actual, message)

        sync_set_op.assert_not_called()
        sync_wait_op.assert_not_called()


if __name__ == "__main__":
    unittest.main()
