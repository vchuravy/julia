; This file is a part of Julia. License is MIT: https://julialang.org/license

; EmitGCStatepoints: calls carrying `julia.gcroots` bundles become
; gc.statepoints with the roots as deopt operands; attributes and the calling
; convention are preserved, and the backend records the deopt locations in the
; .llvm_stackmaps section at the call's return address.

; RUN: opt --load-pass-plugin=libjulia-codegen%shlibext -passes='function(EmitGCStatepoints),verify' -S %s | FileCheck %s
; RUN: opt --load-pass-plugin=libjulia-codegen%shlibext -passes='function(EmitGCStatepoints)' -S %s | llc --load-pass-plugin=libjulia-codegen%shlibext -O2 -mtriple=x86_64-unknown-linux-gnu -use-registers-for-deopt-values -filetype=obj -o - | llvm-readobj --stackmap - | FileCheck %s --check-prefix=STACKMAP

; REQUIRES: x86_64

declare swiftcc ptr @callee_swift(ptr swiftself, ptr)
declare void @throw_it(ptr) noreturn
declare void @use(ptr, ptr, ptr)
declare ptr @box(i64)

; CHECK-LABEL: define ptr @f(ptr swiftself %pgcstack, i64 %a, i64 %b) #{{[0-9]+}} gc "julia"
define ptr @f(ptr swiftself %pgcstack, i64 %a, i64 %b) {
top:
  %aboxed = call ptr @box(i64 %a) [ "julia.gcroots"() ]
  %bboxed = call ptr @box(i64 %b) [ "julia.gcroots"(ptr %aboxed) ]
; CHECK: [[T:%.*]] = call swiftcc token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 {{[0-9]+}}, i32 0, ptr elementtype(ptr (ptr, ptr)) @callee_swift, i32 2, i32 0, ptr swiftself %pgcstack, ptr %aboxed, i32 0, i32 0) [ "deopt"(ptr %aboxed, ptr %bboxed) ]
; CHECK-NEXT: %r = call ptr @llvm.experimental.gc.result.p0(token [[T]])
  %r = call swiftcc ptr @callee_swift(ptr swiftself %pgcstack, ptr %aboxed) [ "julia.gcroots"(ptr %aboxed, ptr %bboxed) ]
  call void @use(ptr %aboxed, ptr %bboxed, ptr %r) [ "julia.gcroots"() ]
  %c = icmp eq i64 %a, 0
  br i1 %c, label %bad, label %ok
bad:
; CHECK: call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 {{[0-9]+}}, i32 0, ptr elementtype(void (ptr)) @throw_it, i32 1, i32 0, ptr %r, i32 0, i32 0) #{{[0-9]+}} [ "deopt"() ]
  call void @throw_it(ptr %r) noreturn [ "julia.gcroots"() ]
  unreachable
ok:
  ret ptr %r
}
; CHECK: attributes #{{[0-9]+}} = { noreturn }

; Three deopt operands across the whole function are recorded; the values live
; across @callee_swift are in callee-saved registers or spill slots, never
; re-spilled by the statepoint itself.
; STACKMAP: LLVM StackMap Version: 3
; STACKMAP: Num Functions: 1
; STACKMAP: Record ID: 3, instruction offset:
; STACKMAP-NEXT: 5 locations:
; STACKMAP-NEXT: #1: Constant 16
; STACKMAP-NEXT: #2: Constant 0
; STACKMAP-NEXT: #3: Constant 2
; STACKMAP-NEXT: #4: Register R#{{[0-9]+}}
; STACKMAP-NEXT: #5: Register R#{{[0-9]+}}
