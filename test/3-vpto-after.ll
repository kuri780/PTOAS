; ============================================================================
; 文件 3：VPTO 后端（改后）— 成功的 LLVM IR 输出
;
; 命令：ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto-llvm-ir kernel.pto
; 输入 PTO IR: pto.print ins("cst=%d\n", %cst : i8)
; 输出：LLVM IR，包含 cce::printf 调用
;
; 改动分支：feature/print-support-research
; 核心改动：在 LowerVPTOOpsPass 中添加 LowerPrintOpPattern，
;           将 pto.print 降级为 LLVM::AddressOfOp + func::CallOp @cce::printf
; ============================================================================

; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

@_ptoas_printf_fmt_0 = private constant [8 x i8] c"cst=%d\0A\00"

declare i32 @"cce::printf"(ptr, i32) #0

define void @vbr_i8_kernel_2d_mix_aiv(ptr addrspace(1) %0) #1 {
  %2 = call i32 @"cce::printf"(ptr @_ptoas_printf_fmt_0, i32 -7)
  ret void
}

attributes #0 = { "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #1 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1, !2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @vbr_i8_kernel_2d_mix_aiv, !"kernel", i32 1}
!2 = !{ptr @vbr_i8_kernel_2d_mix_aiv, !"kernel_with_simd", i32 1}
