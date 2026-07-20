; ============================================================================
; 文件 2：VPTO 后端（改前）— 失败输出
;
; 命令：ptoas --pto-arch=a5 --pto-backend=vpto kernel.pto
; 输入 PTO IR: pto.print ins("cst=%d\n", %cst : i8)
;
; 结果：VPTO 管线在 translateModuleToLLVMIR() 处崩溃，未生成 .ll 文件。
;       错误信息如下：
; ============================================================================

; 错误输出（stderr）:
;   loc("<input>"): error: cannot be converted to LLVM IR: missing
;   `LLVMTranslationDialectInterface` registration for dialect for op:
;   pto.print
;   VPTO LLVM emission failed: LLVM IR export failed for vector module
;   Error: Failed to lower VPTO to LLVM modules.
;
; 根因：LowerVPTOOpsPass 的 populateVPTOOpLoweringPatterns 列出了 ~230 行
;       pattern 覆盖几乎所有 PTO op，但唯独没有 pto::PrintOp 和
;       pto::TPrintOp 的降级 pattern。
;       pto.print 未被任何 pattern 处理，残留到 translateModuleToLLVMIR()，
;       MLIR 不认识 PTO 方言 → 崩溃。
;
; 因此不存在对应的 .ll 文件，这个文件仅记录错误信息。
