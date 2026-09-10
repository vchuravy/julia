; This file is a part of Julia. License is MIT: https://julialang.org/license

; Stackmap GC-root mode (--gc-roots=stackmap): no push/pop of a shadow-stack
; frame; roots live across a safepoint are attached to the call as a
; `julia.gcroots` operand bundle (turned into gc.statepoint deopt operands by
; EmitGCStatepoints); only sret/array slots and roots live across a
; returns_twice call stay in a (null-initialized) memory frame described by a
; whole-frame stackmap record.

; RUN: opt --load-pass-plugin=libjulia-codegen%shlibext -passes='function(LateLowerGCFrameStackmap)' -S %s | FileCheck %s
; RUN: opt --load-pass-plugin=libjulia-codegen%shlibext -passes='function(LateLowerGCFrameStackmap,FinalLowerGCStackmap),verify' -S %s | FileCheck %s --check-prefix=FINAL
; RUN: opt --load-pass-plugin=libjulia-codegen%shlibext -passes='function(LateLowerGCFrameStackmap,FinalLowerGCStackmap,EmitGCStatepoints),verify' -S %s | FileCheck %s --check-prefix=SP

@tag = external addrspace(10) global {}, align 16

declare void @boxed_simple({} addrspace(10)*, {} addrspace(10)*)
declare {} addrspace(10)* @jl_box_int64(i64)
declare {}*** @julia.get_pgcstack()
declare void @julia.safepoint(i64*)
declare i32 @sigsetjmp(i8*, i32) returns_twice
declare void @maybe_throw()
declare void @use({} addrspace(10)*)

define void @register_roots(i64 %a, i64 %b) {
top:
; CHECK-LABEL: @register_roots
; No shadow-stack frame at all: both roots are register resident.
; CHECK-NOT: julia.new_gc_frame
; CHECK-NOT: julia.push_gc_frame
; CHECK-NOT: julia.pop_gc_frame
    %pgcstack = call {}*** @julia.get_pgcstack()
    %aboxed = call {} addrspace(10)* @jl_box_int64(i64 signext %a)
; %aboxed is live across the second call
; CHECK: %bboxed = call ptr addrspace(10) @jl_box_int64(i64 signext %b) [ "julia.gcroots"(ptr addrspace(10) %aboxed) ]
    %bboxed = call {} addrspace(10)* @jl_box_int64(i64 signext %b)
; the arguments of a call are kept alive by the caller for its duration
; CHECK: call void @boxed_simple(ptr addrspace(10) %aboxed, ptr addrspace(10) %bboxed) [ "julia.gcroots"(ptr addrspace(10) %aboxed, ptr addrspace(10) %bboxed) ]
    call void @boxed_simple({} addrspace(10)* %aboxed, {} addrspace(10)* %bboxed)
    ret void

; FINAL-LABEL: @register_roots
; FINAL-NOT: alloca
; FINAL-NOT: llvm.experimental.stackmap

; SP-LABEL: @register_roots
; SP: [[TOK:%.*]] = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 {{[0-9]+}}, i32 0, ptr elementtype(ptr addrspace(10) (i64)) @jl_box_int64, i32 1, i32 0, i64 signext %b, i32 0, i32 0) [ "deopt"(ptr addrspace(10) %aboxed) ]
; SP-NEXT: %bboxed = call ptr addrspace(10) @llvm.experimental.gc.result.p10(token [[TOK]])
; SP: call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 {{[0-9]+}}, i32 0, ptr elementtype(void (ptr addrspace(10), ptr addrspace(10))) @boxed_simple, i32 2, i32 0, ptr addrspace(10) %aboxed, ptr addrspace(10) %bboxed, i32 0, i32 0) [ "deopt"(ptr addrspace(10) %aboxed, ptr addrspace(10) %bboxed) ]
}

define void @safepoint_poll(i64 %a, i64* %page) {
top:
; CHECK-LABEL: @safepoint_poll
    %pgcstack = call {}*** @julia.get_pgcstack()
    %aboxed = call {} addrspace(10)* @jl_box_int64(i64 signext %a)
; CHECK: call void @julia.safepoint(ptr %page) [ "julia.gcroots"(ptr addrspace(10) %aboxed) ]
    call void @julia.safepoint(i64* %page)
; CHECK: call void @use(ptr addrspace(10) %aboxed) [ "julia.gcroots"(ptr addrspace(10) %aboxed) ]
    call void @use({} addrspace(10)* %aboxed)
    ret void
; FINAL-LABEL: @safepoint_poll
; the poll becomes an out-of-line call so that it gets an exact record
; FINAL: call void @jl_gc_safepoint_poll(ptr %page) [ "julia.gcroots"(ptr addrspace(10) %aboxed) ]
; SP-LABEL: @safepoint_poll
; SP: @llvm.experimental.gc.statepoint.p0(i64 {{[0-9]+}}, i32 0, ptr elementtype(void (ptr)) @jl_gc_safepoint_poll, i32 1, i32 0, ptr %page, i32 0, i32 0) [ "deopt"(ptr addrspace(10) %aboxed) ]
}

define void @returns_twice_roots(i64 %a, i8* %jmpbuf) {
top:
; CHECK-LABEL: @returns_twice_roots
; %aboxed is live across the setjmp: it gets a dedicated memory slot in a
; frame of exactly one root, which is never pushed onto the shadow stack.
; CHECK: %gcframe = call ptr @julia.new_gc_frame(i32 1)
; CHECK-NOT: julia.push_gc_frame
    %pgcstack = call {}*** @julia.get_pgcstack()
    %aboxed = call {} addrspace(10)* @jl_box_int64(i64 signext %a)
; CHECK: [[SLOT:%.*]] = call ptr @julia.get_gc_frame_slot(ptr %gcframe, i32 0)
; CHECK-NEXT: store ptr addrspace(10) %aboxed, ptr [[SLOT]]
; CHECK-NEXT: %r = call i32 @sigsetjmp(ptr %jmpbuf, i32 0)
    %r = call i32 @sigsetjmp(i8* %jmpbuf, i32 0) returns_twice
    %c = icmp eq i32 %r, 0
    br i1 %c, label %body, label %catch
body:
; the memory-resident root is not repeated in the bundle
; CHECK: call void @maybe_throw() [ "julia.gcroots"() ]
    call void @maybe_throw()
    ret void
catch:
; CHECK: call void @use(ptr addrspace(10) %aboxed) [ "julia.gcroots"() ]
    call void @use({} addrspace(10)* %aboxed)
; CHECK-NOT: julia.pop_gc_frame
    ret void
; FINAL-LABEL: @returns_twice_roots
; FINAL: %gcframe = alloca ptr addrspace(10), i32 3, align 16
; FINAL-NEXT: call void @llvm.memset.p0.i64(ptr align 16 %gcframe, i8 0, i64 24, i1 false)
; FINAL-NEXT: call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 0, ptr %gcframe, i64 1)
; FINAL-NOT: task.gcstack
}
