; RUN: opt -load libjulia%shlibext -FinalLowerGC -S %s | FileCheck %s

%jl_value_t = type opaque
@tag = external addrspace(10) global %jl_value_t

declare void @boxed_simple(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)
declare %jl_value_t addrspace(10)* @jl_box_int64(i64)
declare %jl_value_t*** @julia.ptls_states()
declare void @jl_safepoint()
declare %jl_value_t addrspace(10)* @jl_apply_generic(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32)

declare noalias nonnull %jl_value_t addrspace(10)** @julia.new_gc_frame(i32)
declare void @julia.push_gc_frame(%jl_value_t addrspace(10)**, i32)
declare %jl_value_t addrspace(10)** @julia.get_gc_frame_slot(%jl_value_t addrspace(10)**, i32)
declare void @julia.pop_gc_frame(%jl_value_t addrspace(10)**)
declare noalias nonnull %jl_value_t addrspace(10)* @julia.gc_alloc_bytes(i8*, i64) #0
declare %jl_value_t* @julia.pointer_from_objref(%jl_value_t addrspace(11)*)
declare token @llvm.julia.gc_preserve_begin(...)
declare void @llvm.julia.gc_preserve_end(token)

attributes #0 = { allocsize(1) }

define void @gc_frame_lowering(i64 %a, i64 %b) {
top:
; CHECK-LABEL: @gc_frame_lowering
; CHECK: %gcframe = alloca %jl_value_t addrspace(10)*, i32 4
  %gcframe = call %jl_value_t addrspace(10)** @julia.new_gc_frame(i32 2)
; CHECK: %ptls = call %jl_value_t*** @julia.ptls_states()
  %ptls = call %jl_value_t*** @julia.ptls_states()
; CHECK-DAG: [[GCFRAME_SIZE_PTR:%.*]] = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %gcframe, i32 0
; CHECK-DAG: [[GCFRAME_SIZE_PTR2:%.*]] = bitcast %jl_value_t addrspace(10)** [[GCFRAME_SIZE_PTR]] to i64*
; CHECK-DAG: store i64 8, i64* [[GCFRAME_SIZE_PTR2]], !tbaa !0
; CHECK-DAG: [[GCFRAME_SLOT:%.*]] = getelementptr %jl_value_t**, %jl_value_t*** %ptls, i32 0
; CHECK-DAG: [[PREV_GCFRAME_PTR:%.*]] = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %gcframe, i32 1
; CHECK-DAG: [[PREV_GCFRAME_PTR2:%.*]] = bitcast %jl_value_t addrspace(10)** [[PREV_GCFRAME_PTR]] to %jl_value_t***
; CHECK-DAG: [[PREV_GCFRAME:%.*]] = load %jl_value_t**, %jl_value_t*** [[GCFRAME_SLOT]]
; CHECK-DAG: store %jl_value_t** [[PREV_GCFRAME]], %jl_value_t*** [[PREV_GCFRAME_PTR2]], !tbaa !0
; CHECK-DAG: [[GCFRAME_SLOT2:%.*]] = bitcast %jl_value_t*** [[GCFRAME_SLOT]] to %jl_value_t addrspace(10)***
; CHECK-NEXT: store %jl_value_t addrspace(10)** %gcframe, %jl_value_t addrspace(10)*** [[GCFRAME_SLOT2]]
  call void @julia.push_gc_frame(%jl_value_t addrspace(10)** %gcframe, i32 2)
  %aboxed = call %jl_value_t addrspace(10)* @jl_box_int64(i64 signext %a)
; CHECK: %frame_slot_1 = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %gcframe, i32 3
  %frame_slot_1 = call %jl_value_t addrspace(10)** @julia.get_gc_frame_slot(%jl_value_t addrspace(10)** %gcframe, i32 1)
  store %jl_value_t addrspace(10)* %aboxed, %jl_value_t addrspace(10)** %frame_slot_1
  %bboxed = call %jl_value_t addrspace(10)* @jl_box_int64(i64 signext %b)
; CHECK: %frame_slot_2 = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %gcframe, i32 2
  %frame_slot_2 = call %jl_value_t addrspace(10)** @julia.get_gc_frame_slot(%jl_value_t addrspace(10)** %gcframe, i32 0)
  store %jl_value_t addrspace(10)* %bboxed, %jl_value_t addrspace(10)** %frame_slot_2
; CHECK: call void @boxed_simple(%jl_value_t addrspace(10)* %aboxed, %jl_value_t addrspace(10)* %bboxed)
  call void @boxed_simple(%jl_value_t addrspace(10)* %aboxed, %jl_value_t addrspace(10)* %bboxed)
; CHECK-NEXT: [[PREV_GCFRAME_PTR3:%.*]] = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %gcframe, i32 1
; CHECK-NEXT: [[PREV_GCFRAME_PTR4:%.*]] = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** [[PREV_GCFRAME_PTR3]], !tbaa !0
; CHECK-NEXT: [[GCFRAME_SLOT3:%.*]] = getelementptr %jl_value_t**, %jl_value_t*** %ptls, i32 0
; CHECK-NEXT: [[GCFRAME_SLOT4:%.*]] = bitcast %jl_value_t*** [[GCFRAME_SLOT3]] to %jl_value_t addrspace(10)**
; CHECK-NEXT: store %jl_value_t addrspace(10)* [[PREV_GCFRAME_PTR4]], %jl_value_t addrspace(10)** [[GCFRAME_SLOT4]], !tbaa !0
  call void @julia.pop_gc_frame(%jl_value_t addrspace(10)** %gcframe)
; CHECK-NEXT: ret void
  ret void
}

define %jl_value_t addrspace(10)* @gc_alloc_lowering() {
top:
; CHECK-LABEL: @gc_alloc_lowering
  %ptls = call %jl_value_t*** @julia.ptls_states()
  %ptls_i8 = bitcast %jl_value_t*** %ptls to i8*
; CHECK: %v = call noalias nonnull %jl_value_t addrspace(10)* @jl_gc_pool_alloc
  %v = call %jl_value_t addrspace(10)* @julia.gc_alloc_bytes(i8* %ptls_i8, i64 8)
  %0 = bitcast %jl_value_t addrspace(10)* %v to %jl_value_t addrspace(10)* addrspace(10)*
  %1 = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)* addrspace(10)* %0, i64 -1
  store %jl_value_t addrspace(10)* @tag, %jl_value_t addrspace(10)* addrspace(10)* %1, !tbaa !0
  ret %jl_value_t addrspace(10)* %v
}

define void @bundle(i8* %fptr) {
; CHECK-LABEL: @bundle
  %ptls = call %jl_value_t*** @julia.ptls_states()
  %ptls_i8 = bitcast %jl_value_t*** %ptls to i8*
  %v = call %jl_value_t addrspace(10)* @julia.gc_alloc_bytes(i8* %ptls_i8, i64 8)
  %va = addrspacecast %jl_value_t addrspace(10)* %v to %jl_value_t addrspace(11)*
  %ptrj = call %jl_value_t* @julia.pointer_from_objref(%jl_value_t addrspace(11)* %va)
  %ptr = bitcast %jl_value_t* %ptrj to i8*
  %f = bitcast i8* %fptr to void (i8*)*
; CHECK: call void %f(i8* %ptr) [ "unknown_bundle"(i8* %ptr) ]
  call void %f(i8* %ptr) [ "jl_roots"(%jl_value_t addrspace(10)* %v), "unknown_bundle"(i8* %ptr) ]
  ret void
}

define void @left_over_gc_preserve() {
; CHECK-LABEL: @left_over_gc_preserve
  %ptls = call %jl_value_t*** @julia.ptls_states()
; CHECK-NOT: @llvm.julia.gc_preserve_begin
; CHECK-NOT: @llvm.julia.gc_preserve_end
  %1 = call token (...) @llvm.julia.gc_preserve_begin()
  call void @llvm.julia.gc_preserve_end(token %1)
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"jtbaa_gcframe", !2, i64 0}
!2 = !{!"jtbaa"}
