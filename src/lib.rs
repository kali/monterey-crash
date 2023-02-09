#![crate_name = "ndarray"]
#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::manual_map,
    clippy::while_let_on_iterator,
    clippy::from_iter_instead_of_collect,
    clippy::redundant_closure
)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
#[cfg(not(feature = "std"))]
extern crate core as std;
pub use crate::arraytraits::AsArray;
pub use crate::dimension::dim::*;
pub use crate::dimension::IxDynImpl;
pub use crate::dimension::NdIndex;
pub use crate::dimension::{Axis, AxisDescription, Dimension, IntoDimension, RemoveAxis};
pub use crate::dimension::{DimAdd, DimMax};
pub use crate::error::{ErrorKind, ShapeError};
pub use crate::impl_views::IndexLonger;
pub use crate::indexes::{indices, indices_of};
use crate::iterators::Baseiter;
use crate::iterators::{ElementsBase, ElementsBaseMut, Iter, IterMut};
pub use crate::linalg_traits::LinalgScalar;
#[cfg(feature = "std")]
pub use crate::linalg_traits::NdFloat;
pub use crate::order::Order;
pub use crate::shape_builder::{Shape, ShapeArg, ShapeBuilder, StrideShape};
pub use crate::slice::{
    MultiSliceArg, NewAxis, Slice, SliceArg, SliceInfo, SliceInfoElem, SliceNextDim,
};
use alloc::sync::Arc;
use std::marker::PhantomData;
#[macro_use]
mod macro_utils {
    macro_rules ! copy_and_clone { ([$ ($ parm : tt) *] $ type_ : ty) => { impl <$ ($ parm) *> Copy for $ type_ { } impl <$ ($ parm) *> Clone for $ type_ { # [inline (always)] fn clone (& self) -> Self { * self } } } ; ($ type_ : ty) => { copy_and_clone ! { [] $ type_ } } }
    macro_rules ! clone_bounds { ([$ ($ parmbounds : tt) *] $ typename : ident [$ ($ parm : tt) *] { @ copy { $ ($ copyfield : ident ,) * } $ ($ field : ident ,) * }) => { impl <$ ($ parmbounds) *> Clone for $ typename <$ ($ parm) *> { fn clone (& self) -> Self { $ typename { $ ($ copyfield : self .$ copyfield ,) * $ ($ field : self .$ field . clone () ,) * } } } } ; }
    #[cfg(debug_assertions)]
    macro_rules ! ndassert { ($ e : expr , $ ($ t : tt) *) => { assert ! ($ e , $ ($ t) *) } }
    #[cfg(not(debug_assertions))]
    macro_rules! ndassert {
        ($ e : expr , $ ($ _ignore : tt) *) => {
            assert!($e)
        };
    }
    macro_rules ! expand_if { (@ bool [true] $ ($ body : tt) *) => { $ ($ body) * } ; (@ bool [false] $ ($ body : tt) *) => { } ; (@ nonempty [$ ($ if_present : tt) +] $ ($ body : tt) *) => { $ ($ body) * } ; (@ nonempty [] $ ($ body : tt) *) => { } ; }
    #[cfg(debug_assertions)]
    macro_rules! debug_bounds_check {
        ($ self_ : ident , $ index : expr) => {
            if $index.index_checked(&$self_.dim, &$self_.strides).is_none() {
                panic!(
                    "ndarray: index {:?} is out of bounds for array of shape {:?}",
                    $index,
                    $self_.shape()
                );
            }
        };
    }
    #[cfg(not(debug_assertions))]
    macro_rules! debug_bounds_check {
        ($ self_ : ident , $ index : expr) => {};
    }
}
#[macro_use]
mod private {
    pub struct PrivateMarker;
    macro_rules! private_decl {
        () => {
            #[doc = " This trait is private to implement; this method exists to make it"]
            #[doc = " impossible to implement outside the crate."]
            #[doc(hidden)]
            fn __private__(&self) -> crate::private::PrivateMarker;
        };
    }
    macro_rules! private_impl {
        () => {
            fn __private__(&self) -> crate::private::PrivateMarker {
                crate::private::PrivateMarker
            }
        };
    }
}
mod aliases {
    use crate::dimension::Dim;
    use crate::{ArcArray, Array, ArrayView, ArrayViewMut, Ix, IxDynImpl};
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn Ix0() -> Ix0 {
        Dim::new([])
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn Ix1(i0: Ix) -> Ix1 {
        Dim::new([i0])
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn Ix2(i0: Ix, i1: Ix) -> Ix2 {
        Dim::new([i0, i1])
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn Ix3(i0: Ix, i1: Ix, i2: Ix) -> Ix3 {
        Dim::new([i0, i1, i2])
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn IxDyn(ix: &[Ix]) -> IxDyn {
        Dim(ix)
    }
    pub type Ix0 = Dim<[Ix; 0]>;
    pub type Ix1 = Dim<[Ix; 1]>;
    pub type Ix2 = Dim<[Ix; 2]>;
    pub type Ix3 = Dim<[Ix; 3]>;
    pub type Ix4 = Dim<[Ix; 4]>;
    pub type Ix5 = Dim<[Ix; 5]>;
    pub type Ix6 = Dim<[Ix; 6]>;
    pub type IxDyn = Dim<IxDynImpl>;
    pub type Array0<A> = Array<A, Ix0>;
    pub type Array1<A> = Array<A, Ix1>;
    pub type Array2<A> = Array<A, Ix2>;
    pub type Array3<A> = Array<A, Ix3>;
    pub type Array4<A> = Array<A, Ix4>;
    pub type Array5<A> = Array<A, Ix5>;
    pub type Array6<A> = Array<A, Ix6>;
    pub type ArrayD<A> = Array<A, IxDyn>;
    pub type ArrayView0<'a, A> = ArrayView<'a, A, Ix0>;
    pub type ArrayView1<'a, A> = ArrayView<'a, A, Ix1>;
    pub type ArrayView2<'a, A> = ArrayView<'a, A, Ix2>;
    pub type ArrayView3<'a, A> = ArrayView<'a, A, Ix3>;
    pub type ArrayView4<'a, A> = ArrayView<'a, A, Ix4>;
    pub type ArrayView5<'a, A> = ArrayView<'a, A, Ix5>;
    pub type ArrayView6<'a, A> = ArrayView<'a, A, Ix6>;
    pub type ArrayViewD<'a, A> = ArrayView<'a, A, IxDyn>;
    pub type ArrayViewMut0<'a, A> = ArrayViewMut<'a, A, Ix0>;
    pub type ArrayViewMut1<'a, A> = ArrayViewMut<'a, A, Ix1>;
    pub type ArrayViewMut2<'a, A> = ArrayViewMut<'a, A, Ix2>;
    pub type ArrayViewMut3<'a, A> = ArrayViewMut<'a, A, Ix3>;
    pub type ArrayViewMut4<'a, A> = ArrayViewMut<'a, A, Ix4>;
    pub type ArrayViewMut5<'a, A> = ArrayViewMut<'a, A, Ix5>;
    pub type ArrayViewMut6<'a, A> = ArrayViewMut<'a, A, Ix6>;
    pub type ArrayViewMutD<'a, A> = ArrayViewMut<'a, A, IxDyn>;
    pub type ArcArray1<A> = ArcArray<A, Ix1>;
    pub type ArcArray2<A> = ArcArray<A, Ix2>;
}
#[macro_use]
mod itertools {
    use std::iter;
    pub(crate) fn enumerate<I>(iterable: I) -> iter::Enumerate<I::IntoIter>
    where
        I: IntoIterator,
    {
        iterable.into_iter().enumerate()
    }
    pub(crate) fn zip<I, J>(i: I, j: J) -> iter::Zip<I::IntoIter, J::IntoIter>
    where
        I: IntoIterator,
        J: IntoIterator,
    {
        i.into_iter().zip(j)
    }
    macro_rules ! izip { (@ closure $ p : pat => $ tup : expr) => { |$ p | $ tup } ; (@ closure $ p : pat => ($ ($ tup : tt) *) , $ _iter : expr $ (, $ tail : expr) *) => { izip ! (@ closure ($ p , b) => ($ ($ tup) *, b) $ (, $ tail) *) } ; ($ first : expr $ (,) *) => { IntoIterator :: into_iter ($ first) } ; ($ first : expr , $ second : expr $ (,) *) => { izip ! ($ first) . zip ($ second) } ; ($ first : expr $ (, $ rest : expr) * $ (,) *) => { izip ! ($ first) $ (. zip ($ rest)) * . map (izip ! (@ closure a => (a) $ (, $ rest) *)) } ; }
}
mod argument_traits {
    use crate::math_cell::MathCell;
    use std::cell::Cell;
    use std::mem::MaybeUninit;
    pub trait AssignElem<T> {
        fn assign_elem(self, input: T);
    }
}
mod arraytraits {
    use crate::imp_prelude::*;
    use crate::{
        dimension,
        iter::{Iter, IterMut},
        numeric_util, FoldWhile, NdIndex, Zip,
    };
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use std::mem;
    use std::ops::{Index, IndexMut};
    use std::{hash, mem::size_of};
    use std::{iter::FromIterator, slice};
    #[cold]
    #[inline(never)]
    pub(crate) fn array_out_of_bounds() -> ! {
        panic!("ndarray: index out of bounds");
    }
    #[inline(always)]
    pub fn debug_bounds_check<S, D, I>(_a: &ArrayBase<S, D>, _index: &I)
    where
        D: Dimension,
        I: NdIndex<D>,
        S: Data,
    {
        debug_bounds_check!(_a, *_index);
    }
    impl<S, D, I> Index<I> for ArrayBase<S, D>
    where
        D: Dimension,
        I: NdIndex<D>,
        S: Data,
    {
        type Output = S::Elem;
        #[inline]
        fn index(&self, index: I) -> &S::Elem {
            debug_bounds_check!(self, index);
            unsafe {
                &*self.ptr.as_ptr().offset(
                    index
                        .index_checked(&self.dim, &self.strides)
                        .unwrap_or_else(|| array_out_of_bounds()),
                )
            }
        }
    }
    impl<A, B, S, S2, D> PartialEq<ArrayBase<S2, D>> for ArrayBase<S, D>
    where
        A: PartialEq<B>,
        S: Data<Elem = A>,
        S2: Data<Elem = B>,
        D: Dimension,
    {
        fn eq(&self, rhs: &ArrayBase<S2, D>) -> bool {
            if self.shape() != rhs.shape() {
                return false;
            }
            if let Some(self_s) = self.as_slice() {
                if let Some(rhs_s) = rhs.as_slice() {
                    return numeric_util::unrolled_eq(self_s, rhs_s);
                }
            }
            Zip::from(self)
                .and(rhs)
                .fold_while(true, |_, a, b| {
                    if a != b {
                        FoldWhile::Done(false)
                    } else {
                        FoldWhile::Continue(true)
                    }
                })
                .into_inner()
        }
    }
    impl<A, S> From<Vec<A>> for ArrayBase<S, Ix1>
    where
        S: DataOwned<Elem = A>,
    {
        fn from(v: Vec<A>) -> Self {
            Self::from_vec(v)
        }
    }
    impl<'a, A, D> IntoIterator for ArrayView<'a, A, D>
    where
        D: Dimension,
    {
        type Item = &'a A;
        type IntoIter = Iter<'a, A, D>;
        fn into_iter(self) -> Self::IntoIter {
            self.into_iter_()
        }
    }
    impl<'a, A, D> IntoIterator for ArrayViewMut<'a, A, D>
    where
        D: Dimension,
    {
        type Item = &'a mut A;
        type IntoIter = IterMut<'a, A, D>;
        fn into_iter(self) -> Self::IntoIter {
            self.into_iter_()
        }
    }
    impl<'a, A, Slice: ?Sized> From<&'a Slice> for ArrayView<'a, A, Ix1>
    where
        Slice: AsRef<[A]>,
    {
        fn from(slice: &'a Slice) -> Self {
            let xs = slice.as_ref();
            if mem::size_of::<A>() == 0 {
                assert!(
                    xs.len() <= ::std::isize::MAX as usize,
                    "Slice length must fit in `isize`.",
                );
            }
            unsafe { Self::from_shape_ptr(xs.len(), xs.as_ptr()) }
        }
    }
    impl<'a, A, const N: usize> From<&'a [[A; N]]> for ArrayView<'a, A, Ix2> {
        fn from(xs: &'a [[A; N]]) -> Self {
            let cols = N;
            let rows = xs.len();
            let dim = Ix2(rows, cols);
            if size_of::<A>() == 0 {
                dimension::size_of_shape_checked(&dim)
                    .expect("Product of non-zero axis lengths must not overflow isize.");
            }
            unsafe {
                let data = slice::from_raw_parts(xs.as_ptr() as *const A, cols * rows);
                ArrayView::from_shape_ptr(dim, data.as_ptr())
            }
        }
    }
    impl<'a, A, Slice: ?Sized> From<&'a mut Slice> for ArrayViewMut<'a, A, Ix1>
    where
        Slice: AsMut<[A]>,
    {
        fn from(slice: &'a mut Slice) -> Self {
            let xs = slice.as_mut();
            if mem::size_of::<A>() == 0 {
                assert!(
                    xs.len() <= ::std::isize::MAX as usize,
                    "Slice length must fit in `isize`.",
                );
            }
            unsafe { Self::from_shape_ptr(xs.len(), xs.as_mut_ptr()) }
        }
    }
    impl<'a, A, const N: usize> From<&'a mut [[A; N]]> for ArrayViewMut<'a, A, Ix2> {
        fn from(xs: &'a mut [[A; N]]) -> Self {
            let cols = N;
            let rows = xs.len();
            let dim = Ix2(rows, cols);
            if size_of::<A>() == 0 {
                dimension::size_of_shape_checked(&dim)
                    .expect("Product of non-zero axis lengths must not overflow isize.");
            }
            unsafe {
                let data = slice::from_raw_parts_mut(xs.as_mut_ptr() as *mut A, cols * rows);
                ArrayViewMut::from_shape_ptr(dim, data.as_mut_ptr())
            }
        }
    }
    pub trait AsArray<'a, A: 'a, D = Ix1>: Into<ArrayView<'a, A, D>>
    where
        D: Dimension,
    {
    }
}
pub use crate::argument_traits::AssignElem;
mod data_repr {
    use crate::extension::nonnull;
    use alloc::borrow::ToOwned;
    use alloc::slice;
    use alloc::vec::Vec;
    use rawpointer::PointerExt;
    use std::mem;
    use std::mem::ManuallyDrop;
    use std::ptr::NonNull;
    #[derive(Debug)]
    #[repr(C)]
    pub struct OwnedRepr<A> {
        ptr: NonNull<A>,
        len: usize,
        capacity: usize,
    }
    impl<A> OwnedRepr<A> {
        pub(crate) fn from(v: Vec<A>) -> Self {
            let mut v = ManuallyDrop::new(v);
            let len = v.len();
            let capacity = v.capacity();
            let ptr = nonnull::nonnull_from_vec_data(&mut v);
            Self { ptr, len, capacity }
        }
        pub(crate) fn as_slice(&self) -> &[A] {
            unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
        pub(crate) fn len(&self) -> usize {
            self.len
        }
        pub(crate) fn as_ptr(&self) -> *const A {
            self.ptr.as_ptr()
        }
        pub(crate) fn as_ptr_mut(&self) -> *mut A {
            self.ptr.as_ptr()
        }
        pub(crate) fn as_nonnull_mut(&mut self) -> NonNull<A> {
            self.ptr
        }
        pub(crate) fn as_end_nonnull(&self) -> NonNull<A> {
            unsafe { self.ptr.add(self.len) }
        }
        #[must_use = "must use new pointer to update existing pointers"]
        pub(crate) fn reserve(&mut self, additional: usize) -> NonNull<A> {
            self.modify_as_vec(|mut v| {
                v.reserve(additional);
                v
            });
            self.as_nonnull_mut()
        }
        pub(crate) unsafe fn set_len(&mut self, new_len: usize) {
            debug_assert!(new_len <= self.capacity);
            self.len = new_len;
        }
        pub(crate) fn release_all_elements(&mut self) -> usize {
            let ret = self.len;
            self.len = 0;
            ret
        }
        fn modify_as_vec(&mut self, f: impl FnOnce(Vec<A>) -> Vec<A>) {
            let v = self.take_as_vec();
            *self = Self::from(f(v));
        }
        fn take_as_vec(&mut self) -> Vec<A> {
            let capacity = self.capacity;
            let len = self.len;
            self.len = 0;
            self.capacity = 0;
            unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), len, capacity) }
        }
    }
    impl<A> Clone for OwnedRepr<A>
    where
        A: Clone,
    {
        fn clone(&self) -> Self {
            Self::from(self.as_slice().to_owned())
        }
    }
    impl<A> Drop for OwnedRepr<A> {
        fn drop(&mut self) {
            if self.capacity > 0 {
                if !mem::needs_drop::<A>() {
                    self.len = 0;
                }
                self.take_as_vec();
            }
        }
    }
}
mod data_traits {
    use crate::{
        ArcArray, Array, ArrayBase, CowRepr, Dimension, OwnedArcRepr, OwnedRepr, RawViewRepr,
        ViewRepr,
    };
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use rawpointer::PointerExt;
    use std::mem::MaybeUninit;
    use std::mem::{self, size_of};
    use std::ptr::NonNull;
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait RawData: Sized {
        type Elem;
        #[deprecated(note = "Unused", since = "0.15.2")]
        fn _data_slice(&self) -> Option<&[Self::Elem]>;
        fn _is_pointer_inbounds(&self, ptr: *const Self::Elem) -> bool;
        private_decl! {}
    }
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait RawDataMut: RawData {
        fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
        where
            Self: Sized,
            D: Dimension;
        fn try_is_unique(&mut self) -> Option<bool>;
    }
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait RawDataClone: RawData {
        unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>);
        unsafe fn clone_from_with_ptr(
            &mut self,
            other: &Self,
            ptr: NonNull<Self::Elem>,
        ) -> NonNull<Self::Elem> {
            let (data, ptr) = other.clone_with_ptr(ptr);
            *self = data;
            ptr
        }
    }
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait Data: RawData {
        #[allow(clippy::wrong_self_convention)]
        fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
        where
            Self::Elem: Clone,
            D: Dimension;
        fn try_into_owned_nocopy<D>(
            self_: ArrayBase<Self, D>,
        ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
        where
            D: Dimension;
        #[allow(clippy::wrong_self_convention)]
        fn to_shared<D>(self_: &ArrayBase<Self, D>) -> ArcArray<Self::Elem, D>
        where
            Self::Elem: Clone,
            D: Dimension,
        {
            self_.to_owned().into_shared()
        }
    }
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait DataMut: Data + RawDataMut {
        #[inline]
        fn ensure_unique<D>(self_: &mut ArrayBase<Self, D>)
        where
            Self: Sized,
            D: Dimension,
        {
            Self::try_ensure_unique(self_)
        }
        #[inline]
        #[allow(clippy::wrong_self_convention)]
        fn is_unique(&mut self) -> bool {
            self.try_is_unique().unwrap()
        }
    }
    unsafe impl<A> RawData for RawViewRepr<*const A> {
        type Elem = A;
        #[inline]
        fn _data_slice(&self) -> Option<&[A]> {
            None
        }
        #[inline(always)]
        fn _is_pointer_inbounds(&self, _ptr: *const Self::Elem) -> bool {
            true
        }
        private_impl! {}
    }
    unsafe impl<A> RawData for RawViewRepr<*mut A> {
        type Elem = A;
        #[inline]
        fn _data_slice(&self) -> Option<&[A]> {
            None
        }
        #[inline(always)]
        fn _is_pointer_inbounds(&self, _ptr: *const Self::Elem) -> bool {
            true
        }
        private_impl! {}
    }
    unsafe impl<A> RawDataClone for RawViewRepr<*mut A> {
        unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
            (*self, ptr)
        }
    }
    unsafe impl<A> RawData for OwnedArcRepr<A> {
        type Elem = A;
        fn _data_slice(&self) -> Option<&[A]> {
            Some(self.0.as_slice())
        }
        fn _is_pointer_inbounds(&self, self_ptr: *const Self::Elem) -> bool {
            self.0._is_pointer_inbounds(self_ptr)
        }
        private_impl! {}
    }
    unsafe impl<A> RawDataMut for OwnedArcRepr<A>
    where
        A: Clone,
    {
        fn try_ensure_unique<D>(self_: &mut ArrayBase<Self, D>)
        where
            Self: Sized,
            D: Dimension,
        {
            if Arc::get_mut(&mut self_.data.0).is_some() {
                return;
            }
            if self_.dim.size() <= self_.data.0.len() / 2 {
                *self_ = self_.to_owned().into_shared();
                return;
            }
            let rcvec = &mut self_.data.0;
            let a_size = mem::size_of::<A>() as isize;
            let our_off = if a_size != 0 {
                (self_.ptr.as_ptr() as isize - rcvec.as_ptr() as isize) / a_size
            } else {
                0
            };
            let rvec = Arc::make_mut(rcvec);
            unsafe {
                self_.ptr = rvec.as_nonnull_mut().offset(our_off);
            }
        }
        fn try_is_unique(&mut self) -> Option<bool> {
            Some(Arc::get_mut(&mut self.0).is_some())
        }
    }
    unsafe impl<A> Data for OwnedArcRepr<A> {
        fn into_owned<D>(mut self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
        where
            A: Clone,
            D: Dimension,
        {
            Self::ensure_unique(&mut self_);
            let data = Arc::try_unwrap(self_.data.0).ok().unwrap();
            unsafe {
                ArrayBase::from_data_ptr(data, self_.ptr).with_strides_dim(self_.strides, self_.dim)
            }
        }
        fn try_into_owned_nocopy<D>(
            self_: ArrayBase<Self, D>,
        ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
        where
            D: Dimension,
        {
            match Arc::try_unwrap(self_.data.0) {
                Ok(owned_data) => unsafe {
                    Ok(ArrayBase::from_data_ptr(owned_data, self_.ptr)
                        .with_strides_dim(self_.strides, self_.dim))
                },
                Err(arc_data) => unsafe {
                    Err(ArrayBase::from_data_ptr(OwnedArcRepr(arc_data), self_.ptr)
                        .with_strides_dim(self_.strides, self_.dim))
                },
            }
        }
    }
    unsafe impl<A> DataMut for OwnedArcRepr<A> where A: Clone {}
    unsafe impl<A> RawDataClone for OwnedArcRepr<A> {
        unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
            (self.clone(), ptr)
        }
    }
    unsafe impl<A> RawData for OwnedRepr<A> {
        type Elem = A;
        fn _data_slice(&self) -> Option<&[A]> {
            Some(self.as_slice())
        }
        fn _is_pointer_inbounds(&self, self_ptr: *const Self::Elem) -> bool {
            let slc = self.as_slice();
            let ptr = slc.as_ptr() as *mut A;
            let end = unsafe { ptr.add(slc.len()) };
            self_ptr >= ptr && self_ptr <= end
        }
        private_impl! {}
    }
    unsafe impl<A> RawDataMut for OwnedRepr<A> {
        #[inline]
        fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
        where
            Self: Sized,
            D: Dimension,
        {
        }
        #[inline]
        fn try_is_unique(&mut self) -> Option<bool> {
            Some(true)
        }
    }
    unsafe impl<A> Data for OwnedRepr<A> {
        #[inline]
        fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
        where
            A: Clone,
            D: Dimension,
        {
            self_
        }
        #[inline]
        fn try_into_owned_nocopy<D>(
            self_: ArrayBase<Self, D>,
        ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
        where
            D: Dimension,
        {
            Ok(self_)
        }
    }
    unsafe impl<A> DataMut for OwnedRepr<A> {}
    unsafe impl<A> RawDataClone for OwnedRepr<A>
    where
        A: Clone,
    {
        unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
            let mut u = self.clone();
            let mut new_ptr = u.as_nonnull_mut();
            if size_of::<A>() != 0 {
                let our_off =
                    (ptr.as_ptr() as isize - self.as_ptr() as isize) / mem::size_of::<A>() as isize;
                new_ptr = new_ptr.offset(our_off);
            }
            (u, new_ptr)
        }
    }
    unsafe impl<'a, A> RawData for ViewRepr<&'a A> {
        type Elem = A;
        #[inline]
        fn _data_slice(&self) -> Option<&[A]> {
            None
        }
        #[inline(always)]
        fn _is_pointer_inbounds(&self, _ptr: *const Self::Elem) -> bool {
            true
        }
        private_impl! {}
    }
    unsafe impl<'a, A> Data for ViewRepr<&'a A> {
        fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
        where
            Self::Elem: Clone,
            D: Dimension,
        {
            self_.to_owned()
        }
        fn try_into_owned_nocopy<D>(
            self_: ArrayBase<Self, D>,
        ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
        where
            D: Dimension,
        {
            Err(self_)
        }
    }
    unsafe impl<'a, A> RawDataClone for ViewRepr<&'a A> {
        unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
            (*self, ptr)
        }
    }
    unsafe impl<'a, A> RawData for ViewRepr<&'a mut A> {
        type Elem = A;
        #[inline]
        fn _data_slice(&self) -> Option<&[A]> {
            None
        }
        #[inline(always)]
        fn _is_pointer_inbounds(&self, _ptr: *const Self::Elem) -> bool {
            true
        }
        private_impl! {}
    }
    unsafe impl<'a, A> RawDataMut for ViewRepr<&'a mut A> {
        #[inline]
        fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
        where
            Self: Sized,
            D: Dimension,
        {
        }
        #[inline]
        fn try_is_unique(&mut self) -> Option<bool> {
            Some(true)
        }
    }
    unsafe impl<'a, A> Data for ViewRepr<&'a mut A> {
        fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
        where
            Self::Elem: Clone,
            D: Dimension,
        {
            self_.to_owned()
        }
        fn try_into_owned_nocopy<D>(
            self_: ArrayBase<Self, D>,
        ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
        where
            D: Dimension,
        {
            Err(self_)
        }
    }
    unsafe impl<'a, A> DataMut for ViewRepr<&'a mut A> {}
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait DataOwned: Data {
        type MaybeUninit: DataOwned<Elem = MaybeUninit<Self::Elem>>
            + RawDataSubst<Self::Elem, Output = Self>;
        fn new(elements: Vec<Self::Elem>) -> Self;
        fn into_shared(self) -> OwnedArcRepr<Self::Elem>;
    }
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait DataShared: Clone + Data + RawDataClone {}
    unsafe impl<A> DataOwned for OwnedRepr<A> {
        type MaybeUninit = OwnedRepr<MaybeUninit<A>>;
        fn new(elements: Vec<A>) -> Self {
            OwnedRepr::from(elements)
        }
        fn into_shared(self) -> OwnedArcRepr<A> {
            OwnedArcRepr(Arc::new(self))
        }
    }
    unsafe impl<'a, A> RawData for CowRepr<'a, A> {
        type Elem = A;
        fn _data_slice(&self) -> Option<&[A]> {
            #[allow(deprecated)]
            match self {
                CowRepr::View(view) => view._data_slice(),
                CowRepr::Owned(data) => data._data_slice(),
            }
        }
        #[inline]
        fn _is_pointer_inbounds(&self, ptr: *const Self::Elem) -> bool {
            match self {
                CowRepr::View(view) => view._is_pointer_inbounds(ptr),
                CowRepr::Owned(data) => data._is_pointer_inbounds(ptr),
            }
        }
        private_impl! {}
    }
    unsafe impl<'a, A> RawDataMut for CowRepr<'a, A>
    where
        A: Clone,
    {
        #[inline]
        fn try_ensure_unique<D>(array: &mut ArrayBase<Self, D>)
        where
            Self: Sized,
            D: Dimension,
        {
            match array.data {
                CowRepr::View(_) => {
                    let owned = array.to_owned();
                    array.data = CowRepr::Owned(owned.data);
                    array.ptr = owned.ptr;
                    array.dim = owned.dim;
                    array.strides = owned.strides;
                }
                CowRepr::Owned(_) => {}
            }
        }
        #[inline]
        fn try_is_unique(&mut self) -> Option<bool> {
            Some(self.is_owned())
        }
    }
    unsafe impl<'a, A> Data for CowRepr<'a, A> {
        #[inline]
        fn into_owned<D>(self_: ArrayBase<CowRepr<'a, A>, D>) -> Array<Self::Elem, D>
        where
            A: Clone,
            D: Dimension,
        {
            match self_.data {
                CowRepr::View(_) => self_.to_owned(),
                CowRepr::Owned(data) => unsafe {
                    ArrayBase::from_data_ptr(data, self_.ptr)
                        .with_strides_dim(self_.strides, self_.dim)
                },
            }
        }
        fn try_into_owned_nocopy<D>(
            self_: ArrayBase<Self, D>,
        ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
        where
            D: Dimension,
        {
            match self_.data {
                CowRepr::View(_) => Err(self_),
                CowRepr::Owned(data) => unsafe {
                    Ok(ArrayBase::from_data_ptr(data, self_.ptr)
                        .with_strides_dim(self_.strides, self_.dim))
                },
            }
        }
    }
    pub trait RawDataSubst<A>: RawData {
        type Output: RawData<Elem = A>;
        unsafe fn data_subst(self) -> Self::Output;
    }
    impl<A, B> RawDataSubst<B> for OwnedRepr<A> {
        type Output = OwnedRepr<B>;
        unsafe fn data_subst(self) -> Self::Output {
            self.data_subst()
        }
    }
    impl<'a, A: 'a, B: 'a> RawDataSubst<B> for ViewRepr<&'a A> {
        type Output = ViewRepr<&'a B>;
        unsafe fn data_subst(self) -> Self::Output {
            ViewRepr::new()
        }
    }
}
pub use crate::aliases::*;
pub use crate::data_traits::{
    Data, DataMut, DataOwned, DataShared, RawData, RawDataClone, RawDataMut, RawDataSubst,
};
mod free_functions {
    use crate::imp_prelude::*;
    use crate::{dimension, ArcArray1, ArcArray2};
    use alloc::vec;
    use alloc::vec::Vec;
    use std::mem::{forget, size_of};
    #[macro_export]
    macro_rules ! array { ($ ([$ ([$ ($ x : expr) ,* $ (,) *]) ,+ $ (,) *]) ,+ $ (,) *) => { { $ crate :: Array3 :: from (vec ! [$ ([$ ([$ ($ x ,) *] ,) *] ,) *]) } } ; ($ ([$ ($ x : expr) ,* $ (,) *]) ,+ $ (,) *) => { { $ crate :: Array2 :: from (vec ! [$ ([$ ($ x ,) *] ,) *]) } } ; ($ ($ x : expr) ,* $ (,) *) => { { $ crate :: Array :: from (vec ! [$ ($ x ,) *]) } } ; }
    pub fn arr0<A>(x: A) -> Array0<A> {
        unsafe { ArrayBase::from_shape_vec_unchecked((), vec![x]) }
    }
    pub fn arr1<A: Clone>(xs: &[A]) -> Array1<A> {
        ArrayBase::from(xs.to_vec())
    }
    pub fn aview0<A>(x: &A) -> ArrayView0<'_, A> {
        unsafe { ArrayView::from_shape_ptr(Ix0(), x) }
    }
    pub fn aview1<A>(xs: &[A]) -> ArrayView1<'_, A> {
        ArrayView::from(xs)
    }
    pub fn aview2<A, const N: usize>(xs: &[[A; N]]) -> ArrayView2<'_, A> {
        ArrayView2::from(xs)
    }
    pub fn aview_mut1<A>(xs: &mut [A]) -> ArrayViewMut1<'_, A> {
        ArrayViewMut::from(xs)
    }
    pub fn arr2<A: Clone, const N: usize>(xs: &[[A; N]]) -> Array2<A> {
        Array2::from(xs.to_vec())
    }
    impl<A, const N: usize> From<Vec<[A; N]>> for Array2<A> {
        fn from(mut xs: Vec<[A; N]>) -> Self {
            let dim = Ix2(xs.len(), N);
            let ptr = xs.as_mut_ptr();
            let cap = xs.capacity();
            let expand_len = dimension::size_of_shape_checked(&dim)
                .expect("Product of non-zero axis lengths must not overflow isize.");
            forget(xs);
            unsafe {
                let v = if size_of::<A>() == 0 {
                    Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
                } else if N == 0 {
                    Vec::new()
                } else {
                    let expand_cap = cap * N;
                    Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
                };
                ArrayBase::from_shape_vec_unchecked(dim, v)
            }
        }
    }
    impl<A, const N: usize, const M: usize> From<Vec<[[A; M]; N]>> for Array3<A> {
        fn from(mut xs: Vec<[[A; M]; N]>) -> Self {
            let dim = Ix3(xs.len(), N, M);
            let ptr = xs.as_mut_ptr();
            let cap = xs.capacity();
            let expand_len = dimension::size_of_shape_checked(&dim)
                .expect("Product of non-zero axis lengths must not overflow isize.");
            forget(xs);
            unsafe {
                let v = if size_of::<A>() == 0 {
                    Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
                } else if N == 0 || M == 0 {
                    Vec::new()
                } else {
                    let expand_cap = cap * N * M;
                    Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
                };
                ArrayBase::from_shape_vec_unchecked(dim, v)
            }
        }
    }
    pub fn arr3<A: Clone, const N: usize, const M: usize>(xs: &[[[A; M]; N]]) -> Array3<A> {
        Array3::from(xs.to_vec())
    }
}
pub use crate::free_functions::*;
pub use crate::iterators::iter;
mod error {
    use super::Dimension;
    #[cfg(feature = "std")]
    use std::error::Error;
    use std::fmt;
    #[derive(Clone)]
    pub struct ShapeError {
        repr: ErrorKind,
    }
    impl ShapeError {
        #[inline]
        pub fn kind(&self) -> ErrorKind {
            self.repr
        }
        pub fn from_kind(error: ErrorKind) -> Self {
            from_kind(error)
        }
    }
    #[non_exhaustive]
    #[derive(Copy, Clone, Debug)]
    pub enum ErrorKind {
        IncompatibleShape = 1,
        IncompatibleLayout,
        RangeLimited,
        OutOfBounds,
        Unsupported,
        Overflow,
    }
    #[inline(always)]
    pub fn from_kind(k: ErrorKind) -> ShapeError {
        ShapeError { repr: k }
    }
    impl PartialEq for ErrorKind {
        #[inline(always)]
        fn eq(&self, rhs: &Self) -> bool {
            *self as u8 == *rhs as u8
        }
    }
    impl PartialEq for ShapeError {
        #[inline(always)]
        fn eq(&self, rhs: &Self) -> bool {
            self.repr == rhs.repr
        }
    }
    impl fmt::Display for ShapeError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let description = match self.kind() {
                ErrorKind::IncompatibleShape => "incompatible shapes",
                ErrorKind::IncompatibleLayout => "incompatible memory layout",
                ErrorKind::RangeLimited => "the shape does not fit in type limits",
                ErrorKind::OutOfBounds => "out of bounds indexing",
                ErrorKind::Unsupported => "unsupported operation",
                ErrorKind::Overflow => "arithmetic overflow",
            };
            write!(f, "ShapeError/{:?}: {}", self.kind(), description)
        }
    }
    impl fmt::Debug for ShapeError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self)
        }
    }
    pub fn incompatible_shapes<D, E>(_a: &D, _b: &E) -> ShapeError
    where
        D: Dimension,
        E: Dimension,
    {
        from_kind(ErrorKind::IncompatibleShape)
    }
}
mod extension {
    pub(crate) mod nonnull {
        use alloc::vec::Vec;
        use std::ptr::NonNull;
        pub(crate) fn nonnull_from_vec_data<T>(v: &mut Vec<T>) -> NonNull<T> {
            unsafe { NonNull::new_unchecked(v.as_mut_ptr()) }
        }
        #[inline]
        pub(crate) unsafe fn nonnull_debug_checked_from_ptr<T>(ptr: *mut T) -> NonNull<T> {
            debug_assert!(!ptr.is_null());
            NonNull::new_unchecked(ptr)
        }
    }
}
mod geomspace {
    #![cfg(feature = "std")]
    use num_traits::Float;
    pub struct Geomspace<F> {
        sign: F,
        start: F,
        step: F,
        index: usize,
        len: usize,
    }
    impl<F> Iterator for Geomspace<F>
    where
        F: Float,
    {
        type Item = F;
        #[inline]
        fn next(&mut self) -> Option<F> {
            if self.index >= self.len {
                None
            } else {
                let i = self.index;
                self.index += 1;
                let exponent = self.start + self.step * F::from(i).unwrap();
                Some(self.sign * exponent.exp())
            }
        }
    }
    impl<F> ExactSizeIterator for Geomspace<F> where Geomspace<F>: Iterator {}
    #[inline]
    pub fn geomspace<F>(a: F, b: F, n: usize) -> Option<Geomspace<F>>
    where
        F: Float,
    {
        if a == F::zero() || b == F::zero() || a.is_sign_negative() != b.is_sign_negative() {
            return None;
        }
        let log_a = a.abs().ln();
        let log_b = b.abs().ln();
        let step = if n > 1 {
            let num_steps =
                F::from(n - 1).expect("Converting number of steps to `A` must not fail.");
            (log_b - log_a) / num_steps
        } else {
            F::zero()
        };
        Some(Geomspace {
            sign: a.signum(),
            start: log_a,
            step,
            index: 0,
            len: n,
        })
    }
}
mod indexes {
    use super::Dimension;
    use crate::dimension::IntoDimension;
    use crate::split_at::SplitAt;
    use crate::zip::Offset;
    use crate::Axis;
    use crate::Layout;
    use crate::NdProducer;
    use crate::{ArrayBase, Data};
    #[derive(Clone)]
    pub struct IndicesIter<D> {
        dim: D,
        index: Option<D>,
    }
    pub fn indices<E>(shape: E) -> Indices<E::Dim>
    where
        E: IntoDimension,
    {
        let dim = shape.into_dimension();
        Indices {
            start: E::Dim::zeros(dim.ndim()),
            dim,
        }
    }
    pub fn indices_of<S, D>(array: &ArrayBase<S, D>) -> Indices<D>
    where
        S: Data,
        D: Dimension,
    {
        indices(array.dim())
    }
    impl<D> Iterator for IndicesIter<D>
    where
        D: Dimension,
    {
        type Item = D::Pattern;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let index = match self.index {
                None => return None,
                Some(ref ix) => ix.clone(),
            };
            self.index = self.dim.next_for(index.clone());
            Some(index.into_pattern())
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            let l = match self.index {
                None => 0,
                Some(ref ix) => {
                    let gone = self
                        .dim
                        .default_strides()
                        .slice()
                        .iter()
                        .zip(ix.slice().iter())
                        .fold(0, |s, (&a, &b)| s + a as usize * b as usize);
                    self.dim.size() - gone
                }
            };
            (l, Some(l))
        }
    }
    impl<D> ExactSizeIterator for IndicesIter<D> where D: Dimension {}
    impl<D> IntoIterator for Indices<D>
    where
        D: Dimension,
    {
        type Item = D::Pattern;
        type IntoIter = IndicesIter<D>;
        fn into_iter(self) -> Self::IntoIter {
            let sz = self.dim.size();
            let index = if sz != 0 { Some(self.start) } else { None };
            IndicesIter {
                index,
                dim: self.dim,
            }
        }
    }
    #[derive(Copy, Clone, Debug)]
    pub struct Indices<D>
    where
        D: Dimension,
    {
        start: D,
        dim: D,
    }
    #[derive(Copy, Clone, Debug)]
    pub struct IndexPtr<D> {
        index: D,
    }
    impl<D> Offset for IndexPtr<D>
    where
        D: Dimension + Copy,
    {
        type Stride = usize;
        unsafe fn stride_offset(mut self, stride: Self::Stride, index: usize) -> Self {
            self.index[stride] += index;
            self
        }
        private_impl! {}
    }
    impl<D: Dimension + Copy> NdProducer for Indices<D> {
        type Item = D::Pattern;
        type Dim = D;
        type Ptr = IndexPtr<D>;
        type Stride = usize;
        private_impl! {}
        fn raw_dim(&self) -> Self::Dim {
            self.dim
        }
        fn as_ptr(&self) -> Self::Ptr {
            IndexPtr { index: self.start }
        }
        fn layout(&self) -> Layout {
            if self.dim.ndim() <= 1 {
                Layout::one_dimensional()
            } else {
                Layout::none()
            }
        }
        unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
            ptr.index.into_pattern()
        }
        unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
            let mut index = *i;
            index += &self.start;
            IndexPtr { index }
        }
        fn stride_of(&self, axis: Axis) -> Self::Stride {
            axis.index()
        }
        #[inline(always)]
        fn contiguous_stride(&self) -> Self::Stride {
            0
        }
        fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
            let start_a = self.start;
            let mut start_b = start_a;
            let (a, b) = self.dim.split_at(axis, index);
            start_b[axis.index()] += index;
            (
                Indices {
                    start: start_a,
                    dim: a,
                },
                Indices {
                    start: start_b,
                    dim: b,
                },
            )
        }
    }
    #[derive(Clone)]
    pub struct IndicesIterF<D> {
        dim: D,
        index: D,
        has_remaining: bool,
    }
    pub fn indices_iter_f<E>(shape: E) -> IndicesIterF<E::Dim>
    where
        E: IntoDimension,
    {
        let dim = shape.into_dimension();
        let zero = E::Dim::zeros(dim.ndim());
        IndicesIterF {
            has_remaining: dim.size_checked() != Some(0),
            index: zero,
            dim,
        }
    }
    impl<D> Iterator for IndicesIterF<D>
    where
        D: Dimension,
    {
        type Item = D::Pattern;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if !self.has_remaining {
                None
            } else {
                let elt = self.index.clone().into_pattern();
                self.has_remaining = self.dim.next_for_f(&mut self.index);
                Some(elt)
            }
        }
    }
    impl<D> ExactSizeIterator for IndicesIterF<D> where D: Dimension {}
}
mod iterators {
    #[macro_use]
    mod macros {
        macro_rules ! impl_ndproducer { ([$ ($ typarm : tt) *] [Clone => $ ($ cloneparm : tt) *] $ typename : ident { $ base : ident , $ ($ fieldname : ident ,) * } $ fulltype : ty { $ (type $ atyn : ident = $ atyv : ty ;) * unsafe fn item (&$ self_ : ident , $ ptr : pat) { $ refexpr : expr } }) => { impl <$ ($ typarm) *> NdProducer for $ fulltype { $ (type $ atyn = $ atyv ;) * type Ptr = * mut A ; type Stride = isize ; fn raw_dim (& self) -> D { self .$ base . raw_dim () } fn layout (& self) -> Layout { self .$ base . layout () } fn as_ptr (& self) -> * mut A { self .$ base . as_ptr () as * mut _ } fn contiguous_stride (& self) -> isize { self .$ base . contiguous_stride () } unsafe fn as_ref (&$ self_ , $ ptr : * mut A) -> Self :: Item { $ refexpr } unsafe fn uget_ptr (& self , i : & Self :: Dim) -> * mut A { self .$ base . uget_ptr (i) } fn stride_of (& self , axis : Axis) -> isize { self .$ base . stride_of (axis) } fn split_at (self , axis : Axis , index : usize) -> (Self , Self) { let (a , b) = self .$ base . split_at (axis , index) ; ($ typename { $ base : a , $ ($ fieldname : self .$ fieldname . clone () ,) * } , $ typename { $ base : b , $ ($ fieldname : self .$ fieldname ,) * }) } private_impl ! { } } expand_if ! (@ nonempty [$ ($ cloneparm) *] impl <$ ($ cloneparm) *> Clone for $ fulltype { fn clone (& self) -> Self { $ typename { $ base : self . base . clone () , $ ($ fieldname : self .$ fieldname . clone () ,) * } } }) ; } }
        macro_rules ! impl_iterator { ([$ ($ typarm : tt) *] [Clone => $ ($ cloneparm : tt) *] $ typename : ident { $ base : ident , $ ($ fieldname : ident ,) * } $ fulltype : ty { type Item = $ ity : ty ; fn item (& mut $ self_ : ident , $ elt : pat) { $ refexpr : expr } }) => { expand_if ! (@ nonempty [$ ($ cloneparm) *] impl <$ ($ cloneparm) *> Clone for $ fulltype { fn clone (& self) -> Self { $ typename { $ base : self .$ base . clone () , $ ($ fieldname : self .$ fieldname . clone () ,) * } } }) ; impl <$ ($ typarm) *> Iterator for $ fulltype { type Item = $ ity ; fn next (& mut $ self_) -> Option < Self :: Item > { $ self_ .$ base . next () . map (|$ elt | { $ refexpr }) } fn size_hint (& self) -> (usize , Option < usize >) { self .$ base . size_hint () } } } }
    }
    mod chunks {
        use crate::imp_prelude::*;
        use crate::ElementsBase;
        use crate::ElementsBaseMut;
        use crate::IntoDimension;
        use crate::{Layout, NdProducer};
        type BaseProducerRef<'a, A, D> = ArrayView<'a, A, D>;
        type BaseProducerMut<'a, A, D> = ArrayViewMut<'a, A, D>;
        pub struct ExactChunks<'a, A, D> {
            base: BaseProducerRef<'a, A, D>,
            chunk: D,
            inner_strides: D,
        }
        impl<'a, A, D: Dimension> ExactChunks<'a, A, D> {}
        pub struct ExactChunksIter<'a, A, D> {
            iter: ElementsBase<'a, A, D>,
            chunk: D,
            inner_strides: D,
        }
        pub struct ExactChunksMut<'a, A, D> {
            base: BaseProducerMut<'a, A, D>,
            chunk: D,
            inner_strides: D,
        }
        impl<'a, A, D: Dimension> ExactChunksMut<'a, A, D> {}
        pub struct ExactChunksIterMut<'a, A, D> {
            iter: ElementsBaseMut<'a, A, D>,
            chunk: D,
            inner_strides: D,
        }
    }
    mod into_iter {
        use super::Baseiter;
        use crate::imp_prelude::*;
        use crate::impl_owned_array::drop_unreachable_raw;
        use crate::OwnedRepr;
        use std::mem;
        use std::ptr::NonNull;
        pub struct IntoIter<A, D>
        where
            D: Dimension,
        {
            array_data: OwnedRepr<A>,
            inner: Baseiter<A, D>,
            data_len: usize,
            array_head_ptr: NonNull<A>,
            has_unreachable_elements: bool,
        }
        impl<A, D> IntoIter<A, D> where D: Dimension {}
        impl<A, D: Dimension> Iterator for IntoIter<A, D> {
            type Item = A;
            #[inline]
            fn next(&mut self) -> Option<A> {
                self.inner.next().map(|p| unsafe { p.read() })
            }
        }
    }
    pub mod iter {
        pub use crate::indexes::{Indices, IndicesIter};
        pub use crate::iterators::{
            AxisChunksIter, AxisChunksIterMut, AxisIter, AxisIterMut, ExactChunks, ExactChunksIter,
            ExactChunksIterMut, ExactChunksMut, IndexedIter, IndexedIterMut, Iter, IterMut, Lanes,
            LanesIter, LanesIterMut, LanesMut, Windows,
        };
    }
    mod lanes {
        use super::LanesIter;
        use super::LanesIterMut;
        use crate::imp_prelude::*;
        use crate::{Layout, NdProducer};
        use std::marker::PhantomData;
        impl_ndproducer! { ['a , A , D : Dimension] [Clone => 'a , A , D : Clone] Lanes { base , inner_len , inner_stride , } Lanes <'a , A , D > { type Item = ArrayView <'a , A , Ix1 >; type Dim = D ; unsafe fn item (& self , ptr) { ArrayView :: new_ (ptr , Ix1 (self . inner_len) , Ix1 (self . inner_stride as Ix)) } } }
        pub struct Lanes<'a, A, D> {
            base: ArrayView<'a, A, D>,
            inner_len: Ix,
            inner_stride: Ixs,
        }
        impl<'a, A, D: Dimension> Lanes<'a, A, D> {
            pub(crate) fn new<Di>(v: ArrayView<'a, A, Di>, axis: Axis) -> Self
            where
                Di: Dimension<Smaller = D>,
            {
                let ndim = v.ndim();
                let len;
                let stride;
                let iter_v = if ndim == 0 {
                    len = 1;
                    stride = 1;
                    v.try_remove_axis(Axis(0))
                } else {
                    let i = axis.index();
                    len = v.dim[i];
                    stride = v.strides[i] as isize;
                    v.try_remove_axis(axis)
                };
                Lanes {
                    inner_len: len,
                    inner_stride: stride,
                    base: iter_v,
                }
            }
        }
        impl_ndproducer! { ['a , A , D : Dimension] [Clone =>] LanesMut { base , inner_len , inner_stride , } LanesMut <'a , A , D > { type Item = ArrayViewMut <'a , A , Ix1 >; type Dim = D ; unsafe fn item (& self , ptr) { ArrayViewMut :: new_ (ptr , Ix1 (self . inner_len) , Ix1 (self . inner_stride as Ix)) } } }
        pub struct LanesMut<'a, A, D> {
            base: ArrayViewMut<'a, A, D>,
            inner_len: Ix,
            inner_stride: Ixs,
        }
        impl<'a, A, D: Dimension> LanesMut<'a, A, D> {
            pub(crate) fn new<Di>(v: ArrayViewMut<'a, A, Di>, axis: Axis) -> Self
            where
                Di: Dimension<Smaller = D>,
            {
                let ndim = v.ndim();
                let len;
                let stride;
                let iter_v = if ndim == 0 {
                    len = 1;
                    stride = 1;
                    v.try_remove_axis(Axis(0))
                } else {
                    let i = axis.index();
                    len = v.dim[i];
                    stride = v.strides[i] as isize;
                    v.try_remove_axis(axis)
                };
                LanesMut {
                    inner_len: len,
                    inner_stride: stride,
                    base: iter_v,
                }
            }
        }
    }
    mod windows {
        use super::ElementsBase;
        use crate::imp_prelude::*;
        use crate::IntoDimension;
        use crate::Layout;
        use crate::NdProducer;
        pub struct Windows<'a, A, D> {
            base: ArrayView<'a, A, D>,
            window: D,
            strides: D,
        }
        impl<'a, A, D: Dimension> Windows<'a, A, D> {}
        pub struct WindowsIter<'a, A, D> {
            iter: ElementsBase<'a, A, D>,
            window: D,
            strides: D,
        }
    }
    pub use self::chunks::{ExactChunks, ExactChunksIter, ExactChunksIterMut, ExactChunksMut};
    pub use self::into_iter::IntoIter;
    pub use self::lanes::{Lanes, LanesMut};
    pub use self::windows::Windows;
    use super::{ArrayBase, ArrayView, ArrayViewMut, Axis, Data, NdProducer, RemoveAxis};
    use super::{Dimension, Ix, Ixs};
    use crate::Ix1;
    use alloc::vec::Vec;
    use std::iter::FromIterator;
    use std::marker::PhantomData;
    use std::ptr;
    use std::slice::{self, Iter as SliceIter, IterMut as SliceIterMut};
    pub struct Baseiter<A, D> {
        ptr: *mut A,
        dim: D,
        strides: D,
        index: Option<D>,
    }
    impl<A, D: Dimension> Baseiter<A, D> {
        #[inline]
        pub unsafe fn new(ptr: *mut A, len: D, stride: D) -> Baseiter<A, D> {
            Baseiter {
                ptr,
                index: len.first_index(),
                dim: len,
                strides: stride,
            }
        }
    }
    impl<A, D: Dimension> Iterator for Baseiter<A, D> {
        type Item = *mut A;
        #[inline]
        fn next(&mut self) -> Option<*mut A> {
            let index = match self.index {
                None => return None,
                Some(ref ix) => ix.clone(),
            };
            let offset = D::stride_offset(&index, &self.strides);
            self.index = self.dim.next_for(index);
            unsafe { Some(self.ptr.offset(offset)) }
        }
    }
    impl<A, D: Dimension> ExactSizeIterator for Baseiter<A, D> {}
    impl<A> DoubleEndedIterator for Baseiter<A, Ix1> {
        #[inline]
        fn next_back(&mut self) -> Option<*mut A> {
            let index = match self.index {
                None => return None,
                Some(ix) => ix,
            };
            self.dim[0] -= 1;
            let offset = <_>::stride_offset(&self.dim, &self.strides);
            if index == self.dim {
                self.index = None;
            }
            unsafe { Some(self.ptr.offset(offset)) }
        }
    }
    clone_bounds ! ([A , D : Clone] Baseiter [A , D] { @ copy { ptr , } dim , strides , index , });
    clone_bounds ! (['a , A , D : Clone] ElementsBase ['a , A , D] { @ copy { life , } inner , });
    impl<'a, A, D: Dimension> ElementsBase<'a, A, D> {
        pub fn new(v: ArrayView<'a, A, D>) -> Self {
            ElementsBase {
                inner: v.into_base_iter(),
                life: PhantomData,
            }
        }
    }
    impl<'a, A, D: Dimension> Iterator for ElementsBase<'a, A, D> {
        type Item = &'a A;
        #[inline]
        fn next(&mut self) -> Option<&'a A> {
            self.inner.next().map(|p| unsafe { &*p })
        }
    }
    impl<'a, A> DoubleEndedIterator for ElementsBase<'a, A, Ix1> {
        #[inline]
        fn next_back(&mut self) -> Option<&'a A> {
            self.inner.next_back().map(|p| unsafe { &*p })
        }
    }
    impl<'a, A, D> ExactSizeIterator for ElementsBase<'a, A, D> where D: Dimension {}
    macro_rules! either {
        ($ value : expr , $ inner : pat => $ result : expr) => {
            match $value {
                ElementsRepr::Slice($inner) => $result,
                ElementsRepr::Counted($inner) => $result,
            }
        };
    }
    macro_rules! either_mut {
        ($ value : expr , $ inner : ident => $ result : expr) => {
            match $value {
                ElementsRepr::Slice(ref mut $inner) => $result,
                ElementsRepr::Counted(ref mut $inner) => $result,
            }
        };
    }
    impl<'a, A, D> Iter<'a, A, D>
    where
        D: Dimension,
    {
        pub(crate) fn new(self_: ArrayView<'a, A, D>) -> Self {
            Iter {
                inner: if let Some(slc) = self_.to_slice() {
                    ElementsRepr::Slice(slc.iter())
                } else {
                    ElementsRepr::Counted(self_.into_elements_base())
                },
            }
        }
    }
    impl<'a, A, D> IterMut<'a, A, D>
    where
        D: Dimension,
    {
        pub(crate) fn new(self_: ArrayViewMut<'a, A, D>) -> Self {
            IterMut {
                inner: match self_.try_into_slice() {
                    Ok(x) => ElementsRepr::Slice(x.iter_mut()),
                    Err(self_) => ElementsRepr::Counted(self_.into_elements_base()),
                },
            }
        }
    }
    #[derive(Clone)]
    pub enum ElementsRepr<S, C> {
        Slice(S),
        Counted(C),
    }
    pub struct Iter<'a, A, D> {
        inner: ElementsRepr<SliceIter<'a, A>, ElementsBase<'a, A, D>>,
    }
    pub struct ElementsBase<'a, A, D> {
        inner: Baseiter<A, D>,
        life: PhantomData<&'a A>,
    }
    pub struct IterMut<'a, A, D> {
        inner: ElementsRepr<SliceIterMut<'a, A>, ElementsBaseMut<'a, A, D>>,
    }
    pub struct ElementsBaseMut<'a, A, D> {
        inner: Baseiter<A, D>,
        life: PhantomData<&'a mut A>,
    }
    impl<'a, A, D: Dimension> ElementsBaseMut<'a, A, D> {
        pub fn new(v: ArrayViewMut<'a, A, D>) -> Self {
            ElementsBaseMut {
                inner: v.into_base_iter(),
                life: PhantomData,
            }
        }
    }
    #[derive(Clone)]
    pub struct IndexedIter<'a, A, D>(ElementsBase<'a, A, D>);
    pub struct IndexedIterMut<'a, A, D>(ElementsBaseMut<'a, A, D>);
    impl<'a, A, D> IndexedIter<'a, A, D>
    where
        D: Dimension,
    {
        pub(crate) fn new(x: ElementsBase<'a, A, D>) -> Self {
            IndexedIter(x)
        }
    }
    impl<'a, A, D> IndexedIterMut<'a, A, D>
    where
        D: Dimension,
    {
        pub(crate) fn new(x: ElementsBaseMut<'a, A, D>) -> Self {
            IndexedIterMut(x)
        }
    }
    impl<'a, A, D: Dimension> Iterator for Iter<'a, A, D> {
        type Item = &'a A;
        #[inline]
        fn next(&mut self) -> Option<&'a A> {
            either_mut ! (self . inner , iter => iter . next ())
        }
    }
    impl<'a, A, D> ExactSizeIterator for Iter<'a, A, D> where D: Dimension {}
    impl<'a, A, D: Dimension> Iterator for IndexedIter<'a, A, D> {
        type Item = (D::Pattern, &'a A);
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let index = match self.0.inner.index {
                None => return None,
                Some(ref ix) => ix.clone(),
            };
            match self.0.next() {
                None => None,
                Some(elem) => Some((index.into_pattern(), elem)),
            }
        }
    }
    impl<'a, A, D: Dimension> Iterator for IterMut<'a, A, D> {
        type Item = &'a mut A;
        #[inline]
        fn next(&mut self) -> Option<&'a mut A> {
            either_mut ! (self . inner , iter => iter . next ())
        }
    }
    impl<'a, A, D> ExactSizeIterator for IterMut<'a, A, D> where D: Dimension {}
    impl<'a, A, D: Dimension> Iterator for ElementsBaseMut<'a, A, D> {
        type Item = &'a mut A;
        #[inline]
        fn next(&mut self) -> Option<&'a mut A> {
            self.inner.next().map(|p| unsafe { &mut *p })
        }
    }
    impl<'a, A, D> ExactSizeIterator for ElementsBaseMut<'a, A, D> where D: Dimension {}
    impl<'a, A, D: Dimension> Iterator for IndexedIterMut<'a, A, D> {
        type Item = (D::Pattern, &'a mut A);
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let index = match self.0.inner.index {
                None => return None,
                Some(ref ix) => ix.clone(),
            };
            match self.0.next() {
                None => None,
                Some(elem) => Some((index.into_pattern(), elem)),
            }
        }
    }
    pub struct LanesIter<'a, A, D> {
        inner_len: Ix,
        inner_stride: Ixs,
        iter: Baseiter<A, D>,
        life: PhantomData<&'a A>,
    }
    impl<'a, A, D> Iterator for LanesIter<'a, A, D>
    where
        D: Dimension,
    {
        type Item = ArrayView<'a, A, Ix1>;
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().map(|ptr| unsafe {
                ArrayView::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
            })
        }
    }
    pub struct LanesIterMut<'a, A, D> {
        inner_len: Ix,
        inner_stride: Ixs,
        iter: Baseiter<A, D>,
        life: PhantomData<&'a mut A>,
    }
    impl<'a, A, D> Iterator for LanesIterMut<'a, A, D>
    where
        D: Dimension,
    {
        type Item = ArrayViewMut<'a, A, Ix1>;
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().map(|ptr| unsafe {
                ArrayViewMut::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
            })
        }
    }
    #[derive(Debug)]
    pub struct AxisIterCore<A, D> {
        index: Ix,
        end: Ix,
        stride: Ixs,
        inner_dim: D,
        inner_strides: D,
        ptr: *mut A,
    }
    clone_bounds ! ([A , D : Clone] AxisIterCore [A , D] { @ copy { index , end , stride , ptr , } inner_dim , inner_strides , });
    impl<A, D: Dimension> AxisIterCore<A, D> {
        fn new<S, Di>(v: ArrayBase<S, Di>, axis: Axis) -> Self
        where
            Di: RemoveAxis<Smaller = D>,
            S: Data<Elem = A>,
        {
            AxisIterCore {
                index: 0,
                end: v.len_of(axis),
                stride: v.stride_of(axis),
                inner_dim: v.dim.remove_axis(axis),
                inner_strides: v.strides.remove_axis(axis),
                ptr: v.ptr.as_ptr(),
            }
        }
        #[inline]
        unsafe fn offset(&self, index: usize) -> *mut A {
            debug_assert!(
                index < self.end,
                "index={}, end={}, stride={}",
                index,
                self.end,
                self.stride
            );
            self.ptr.offset(index as isize * self.stride)
        }
        fn split_at(self, index: usize) -> (Self, Self) {
            assert!(index <= self.len());
            let mid = self.index + index;
            let left = AxisIterCore {
                index: self.index,
                end: mid,
                stride: self.stride,
                inner_dim: self.inner_dim.clone(),
                inner_strides: self.inner_strides.clone(),
                ptr: self.ptr,
            };
            let right = AxisIterCore {
                index: mid,
                end: self.end,
                stride: self.stride,
                inner_dim: self.inner_dim,
                inner_strides: self.inner_strides,
                ptr: self.ptr,
            };
            (left, right)
        }
    }
    impl<A, D> Iterator for AxisIterCore<A, D>
    where
        D: Dimension,
    {
        type Item = *mut A;
        fn next(&mut self) -> Option<Self::Item> {
            if self.index >= self.end {
                None
            } else {
                let ptr = unsafe { self.offset(self.index) };
                self.index += 1;
                Some(ptr)
            }
        }
    }
    impl<A, D> DoubleEndedIterator for AxisIterCore<A, D>
    where
        D: Dimension,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.index >= self.end {
                None
            } else {
                let ptr = unsafe { self.offset(self.end - 1) };
                self.end -= 1;
                Some(ptr)
            }
        }
    }
    impl<A, D> ExactSizeIterator for AxisIterCore<A, D> where D: Dimension {}
    #[derive(Debug)]
    pub struct AxisIter<'a, A, D> {
        iter: AxisIterCore<A, D>,
        life: PhantomData<&'a A>,
    }
    impl<'a, A, D: Dimension> AxisIter<'a, A, D> {
        pub(crate) fn new<Di>(v: ArrayView<'a, A, Di>, axis: Axis) -> Self
        where
            Di: RemoveAxis<Smaller = D>,
        {
            AxisIter {
                iter: AxisIterCore::new(v, axis),
                life: PhantomData,
            }
        }
        pub fn split_at(self, index: usize) -> (Self, Self) {
            let (left, right) = self.iter.split_at(index);
            (
                AxisIter {
                    iter: left,
                    life: self.life,
                },
                AxisIter {
                    iter: right,
                    life: self.life,
                },
            )
        }
    }
    impl<'a, A, D> Iterator for AxisIter<'a, A, D>
    where
        D: Dimension,
    {
        type Item = ArrayView<'a, A, D>;
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().map(|ptr| unsafe { self.as_ref(ptr) })
        }
    }
    impl<'a, A, D> ExactSizeIterator for AxisIter<'a, A, D> where D: Dimension {}
    pub struct AxisIterMut<'a, A, D> {
        iter: AxisIterCore<A, D>,
        life: PhantomData<&'a mut A>,
    }
    impl<'a, A, D: Dimension> AxisIterMut<'a, A, D> {
        pub(crate) fn new<Di>(v: ArrayViewMut<'a, A, Di>, axis: Axis) -> Self
        where
            Di: RemoveAxis<Smaller = D>,
        {
            AxisIterMut {
                iter: AxisIterCore::new(v, axis),
                life: PhantomData,
            }
        }
        pub fn split_at(self, index: usize) -> (Self, Self) {
            let (left, right) = self.iter.split_at(index);
            (
                AxisIterMut {
                    iter: left,
                    life: self.life,
                },
                AxisIterMut {
                    iter: right,
                    life: self.life,
                },
            )
        }
    }
    impl<'a, A, D> Iterator for AxisIterMut<'a, A, D>
    where
        D: Dimension,
    {
        type Item = ArrayViewMut<'a, A, D>;
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().map(|ptr| unsafe { self.as_ref(ptr) })
        }
    }
    impl<'a, A, D> ExactSizeIterator for AxisIterMut<'a, A, D> where D: Dimension {}
    impl<'a, A, D: Dimension> NdProducer for AxisIter<'a, A, D> {
        type Item = <Self as Iterator>::Item;
        type Dim = Ix1;
        type Ptr = *mut A;
        type Stride = isize;
        fn layout(&self) -> crate::Layout {
            crate::Layout::one_dimensional()
        }
        fn raw_dim(&self) -> Self::Dim {
            Ix1(self.len())
        }
        fn as_ptr(&self) -> Self::Ptr {
            if self.len() > 0 {
                unsafe { self.iter.offset(self.iter.index) }
            } else {
                std::ptr::NonNull::dangling().as_ptr()
            }
        }
        fn contiguous_stride(&self) -> isize {
            self.iter.stride
        }
        unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
            ArrayView::new_(
                ptr,
                self.iter.inner_dim.clone(),
                self.iter.inner_strides.clone(),
            )
        }
        unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
            self.iter.offset(self.iter.index + i[0])
        }
        fn stride_of(&self, _axis: Axis) -> isize {
            self.contiguous_stride()
        }
        fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
            self.split_at(index)
        }
        private_impl! {}
    }
    impl<'a, A, D: Dimension> NdProducer for AxisIterMut<'a, A, D> {
        type Item = <Self as Iterator>::Item;
        type Dim = Ix1;
        type Ptr = *mut A;
        type Stride = isize;
        fn layout(&self) -> crate::Layout {
            crate::Layout::one_dimensional()
        }
        fn raw_dim(&self) -> Self::Dim {
            Ix1(self.len())
        }
        fn as_ptr(&self) -> Self::Ptr {
            if self.len() > 0 {
                unsafe { self.iter.offset(self.iter.index) }
            } else {
                std::ptr::NonNull::dangling().as_ptr()
            }
        }
        fn contiguous_stride(&self) -> isize {
            self.iter.stride
        }
        unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
            ArrayViewMut::new_(
                ptr,
                self.iter.inner_dim.clone(),
                self.iter.inner_strides.clone(),
            )
        }
        unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
            self.iter.offset(self.iter.index + i[0])
        }
        fn stride_of(&self, _axis: Axis) -> isize {
            self.contiguous_stride()
        }
        fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
            self.split_at(index)
        }
        private_impl! {}
    }
    pub struct AxisChunksIter<'a, A, D> {
        iter: AxisIterCore<A, D>,
        partial_chunk_index: usize,
        partial_chunk_dim: D,
        life: PhantomData<&'a A>,
    }
    fn chunk_iter_parts<A, D: Dimension>(
        v: ArrayView<'_, A, D>,
        axis: Axis,
        size: usize,
    ) -> (AxisIterCore<A, D>, usize, D) {
        assert_ne!(size, 0, "Chunk size must be nonzero.");
        let axis_len = v.len_of(axis);
        let n_whole_chunks = axis_len / size;
        let chunk_remainder = axis_len % size;
        let iter_len = if chunk_remainder == 0 {
            n_whole_chunks
        } else {
            n_whole_chunks + 1
        };
        let stride = if n_whole_chunks == 0 {
            0
        } else {
            v.stride_of(axis) * size as isize
        };
        let axis = axis.index();
        let mut inner_dim = v.dim.clone();
        inner_dim[axis] = size;
        let mut partial_chunk_dim = v.dim;
        partial_chunk_dim[axis] = chunk_remainder;
        let partial_chunk_index = n_whole_chunks;
        let iter = AxisIterCore {
            index: 0,
            end: iter_len,
            stride,
            inner_dim,
            inner_strides: v.strides,
            ptr: v.ptr.as_ptr(),
        };
        (iter, partial_chunk_index, partial_chunk_dim)
    }
    impl<'a, A, D: Dimension> AxisChunksIter<'a, A, D> {
        pub(crate) fn new(v: ArrayView<'a, A, D>, axis: Axis, size: usize) -> Self {
            let (iter, partial_chunk_index, partial_chunk_dim) = chunk_iter_parts(v, axis, size);
            AxisChunksIter {
                iter,
                partial_chunk_index,
                partial_chunk_dim,
                life: PhantomData,
            }
        }
    }
    macro_rules! chunk_iter_impl {
        ($ iter : ident , $ array : ident) => {
            impl<'a, A, D> $iter<'a, A, D>
            where
                D: Dimension,
            {
                fn get_subview(&self, index: usize, ptr: *mut A) -> $array<'a, A, D> {
                    if index != self.partial_chunk_index {
                        unsafe {
                            $array::new_(
                                ptr,
                                self.iter.inner_dim.clone(),
                                self.iter.inner_strides.clone(),
                            )
                        }
                    } else {
                        unsafe {
                            $array::new_(
                                ptr,
                                self.partial_chunk_dim.clone(),
                                self.iter.inner_strides.clone(),
                            )
                        }
                    }
                }
                #[doc = " Splits the iterator at index, yielding two disjoint iterators."]
                #[doc = ""]
                #[doc = " `index` is relative to the current state of the iterator (which is not"]
                #[doc = " necessarily the start of the axis)."]
                #[doc = ""]
                #[doc = " **Panics** if `index` is strictly greater than the iterator's remaining"]
                #[doc = " length."]
                pub fn split_at(self, index: usize) -> (Self, Self) {
                    let (left, right) = self.iter.split_at(index);
                    (
                        Self {
                            iter: left,
                            partial_chunk_index: self.partial_chunk_index,
                            partial_chunk_dim: self.partial_chunk_dim.clone(),
                            life: self.life,
                        },
                        Self {
                            iter: right,
                            partial_chunk_index: self.partial_chunk_index,
                            partial_chunk_dim: self.partial_chunk_dim,
                            life: self.life,
                        },
                    )
                }
            }
            impl<'a, A, D> Iterator for $iter<'a, A, D>
            where
                D: Dimension,
            {
                type Item = $array<'a, A, D>;
                fn next(&mut self) -> Option<Self::Item> {
                    self.iter
                        .next_with_index()
                        .map(|(index, ptr)| self.get_subview(index, ptr))
                }
                fn size_hint(&self) -> (usize, Option<usize>) {
                    self.iter.size_hint()
                }
            }
            impl<'a, A, D> DoubleEndedIterator for $iter<'a, A, D>
            where
                D: Dimension,
            {
                fn next_back(&mut self) -> Option<Self::Item> {
                    self.iter
                        .next_back_with_index()
                        .map(|(index, ptr)| self.get_subview(index, ptr))
                }
            }
            impl<'a, A, D> ExactSizeIterator for $iter<'a, A, D> where D: Dimension {}
        };
    }
    pub struct AxisChunksIterMut<'a, A, D> {
        iter: AxisIterCore<A, D>,
        partial_chunk_index: usize,
        partial_chunk_dim: D,
        life: PhantomData<&'a mut A>,
    }
    impl<'a, A, D: Dimension> AxisChunksIterMut<'a, A, D> {
        pub(crate) fn new(v: ArrayViewMut<'a, A, D>, axis: Axis, size: usize) -> Self {
            let (iter, partial_chunk_index, partial_chunk_dim) =
                chunk_iter_parts(v.into_view(), axis, size);
            AxisChunksIterMut {
                iter,
                partial_chunk_index,
                partial_chunk_dim,
                life: PhantomData,
            }
        }
    }
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait TrustedIterator {}
    use crate::indexes::IndicesIterF;
    use crate::iter::IndicesIter;
    #[cfg(feature = "std")]
    use crate::{geomspace::Geomspace, linspace::Linspace, logspace::Logspace};
    #[cfg(feature = "std")]
    unsafe impl<F> TrustedIterator for Linspace<F> {}
    #[cfg(feature = "std")]
    unsafe impl<F> TrustedIterator for Geomspace<F> {}
    #[cfg(feature = "std")]
    unsafe impl<F> TrustedIterator for Logspace<F> {}
    unsafe impl<'a, A, D> TrustedIterator for Iter<'a, A, D> {}
    unsafe impl<'a, A, D> TrustedIterator for IterMut<'a, A, D> {}
    unsafe impl<I> TrustedIterator for std::iter::Cloned<I> where I: TrustedIterator {}
    unsafe impl<'a, A> TrustedIterator for slice::Iter<'a, A> {}
    unsafe impl<'a, A> TrustedIterator for slice::IterMut<'a, A> {}
    unsafe impl TrustedIterator for ::std::ops::Range<usize> {}
    unsafe impl<D> TrustedIterator for IndicesIter<D> where D: Dimension {}
    unsafe impl<D> TrustedIterator for IndicesIterF<D> where D: Dimension {}
    pub fn to_vec<I>(iter: I) -> Vec<I::Item>
    where
        I: TrustedIterator + ExactSizeIterator,
    {
        to_vec_mapped(iter, |x| x)
    }
    pub fn to_vec_mapped<I, F, B>(iter: I, mut f: F) -> Vec<B>
    where
        I: TrustedIterator + ExactSizeIterator,
        F: FnMut(I::Item) -> B,
    {
        let (size, _) = iter.size_hint();
        let mut result = Vec::with_capacity(size);
        let mut out_ptr = result.as_mut_ptr();
        let mut len = 0;
        iter.fold((), |(), elt| unsafe {
            ptr::write(out_ptr, f(elt));
            len += 1;
            result.set_len(len);
            out_ptr = out_ptr.offset(1);
        });
        debug_assert_eq!(size, result.len());
        result
    }
}
mod layout {
    mod layoutfmt {
        use super::Layout;
        const LAYOUT_NAMES: &[&str] = &["C", "F", "c", "f"];
        use std::fmt;
        impl fmt::Debug for Layout {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.0 == 0 {
                    write!(f, "Custom")?
                } else {
                    (0..32).filter(|&i| self.is(1 << i)).try_fold((), |_, i| {
                        if let Some(name) = LAYOUT_NAMES.get(i) {
                            write!(f, "{}", name)
                        } else {
                            write!(f, "{:#x}", i)
                        }
                    })?;
                };
                write!(f, " ({:#x})", self.0)
            }
        }
    }
    #[derive(Copy, Clone)]
    pub struct Layout(u32);
    impl Layout {
        pub(crate) const CORDER: u32 = 0b01;
        pub(crate) const FORDER: u32 = 0b10;
        pub(crate) const CPREFER: u32 = 0b0100;
        pub(crate) const FPREFER: u32 = 0b1000;
        #[inline(always)]
        pub(crate) fn is(self, flag: u32) -> bool {
            self.0 & flag != 0
        }
        #[inline(always)]
        pub(crate) fn intersect(self, other: Layout) -> Layout {
            Layout(self.0 & other.0)
        }
        #[inline(always)]
        pub(crate) fn also(self, other: Layout) -> Layout {
            Layout(self.0 | other.0)
        }
        #[inline(always)]
        pub(crate) fn one_dimensional() -> Layout {
            Layout::c().also(Layout::f())
        }
        #[inline(always)]
        pub(crate) fn c() -> Layout {
            Layout(Layout::CORDER | Layout::CPREFER)
        }
        #[inline(always)]
        pub(crate) fn f() -> Layout {
            Layout(Layout::FORDER | Layout::FPREFER)
        }
        #[inline(always)]
        pub(crate) fn cpref() -> Layout {
            Layout(Layout::CPREFER)
        }
        #[inline(always)]
        pub(crate) fn fpref() -> Layout {
            Layout(Layout::FPREFER)
        }
        #[inline(always)]
        pub(crate) fn none() -> Layout {
            Layout(0)
        }
        #[inline]
        pub(crate) fn tendency(self) -> i32 {
            (self.is(Layout::CORDER) as i32 - self.is(Layout::FORDER) as i32)
                + (self.is(Layout::CPREFER) as i32 - self.is(Layout::FPREFER) as i32)
        }
    }
}
mod linalg_traits {
    #[cfg(feature = "std")]
    use crate::ScalarOperand;
    #[cfg(feature = "std")]
    use num_traits::Float;
    use num_traits::{One, Zero};
    #[cfg(feature = "std")]
    use std::fmt;
    use std::ops::{Add, Div, Mul, Sub};
    #[cfg(feature = "std")]
    use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};
    pub trait LinalgScalar:
        'static
        + Copy
        + Zero
        + One
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
    {
    }
    impl<T> LinalgScalar for T where
        T: 'static
            + Copy
            + Zero
            + One
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
    {
    }
    #[cfg(feature = "std")]
    pub trait NdFloat:
        Float
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + fmt::Display
        + fmt::Debug
        + fmt::LowerExp
        + fmt::UpperExp
        + ScalarOperand
        + LinalgScalar
        + Send
        + Sync
    {
    }
}
mod linspace {
    #![cfg(feature = "std")]
    use num_traits::Float;
    pub struct Linspace<F> {
        start: F,
        step: F,
        index: usize,
        len: usize,
    }
    impl<F> Iterator for Linspace<F>
    where
        F: Float,
    {
        type Item = F;
        #[inline]
        fn next(&mut self) -> Option<F> {
            if self.index >= self.len {
                None
            } else {
                let i = self.index;
                self.index += 1;
                Some(self.start + self.step * F::from(i).unwrap())
            }
        }
    }
    impl<F> ExactSizeIterator for Linspace<F> where Linspace<F>: Iterator {}
    #[inline]
    pub fn linspace<F>(a: F, b: F, n: usize) -> Linspace<F>
    where
        F: Float,
    {
        let step = if n > 1 {
            let num_steps =
                F::from(n - 1).expect("Converting number of steps to `A` must not fail.");
            (b - a) / num_steps
        } else {
            F::zero()
        };
        Linspace {
            start: a,
            step,
            index: 0,
            len: n,
        }
    }
    #[inline]
    pub fn range<F>(a: F, b: F, step: F) -> Linspace<F>
    where
        F: Float,
    {
        let len = b - a;
        let steps = F::ceil(len / step);
        Linspace {
            start: a,
            step,
            len: steps.to_usize().expect(
                "Converting the length to `usize` must not fail. The most likely \
             cause of this failure is if the sign of `end - start` is \
             different from the sign of `step`.",
            ),
            index: 0,
        }
    }
}
mod logspace {
    #![cfg(feature = "std")]
    use num_traits::Float;
    pub struct Logspace<F> {
        sign: F,
        base: F,
        start: F,
        step: F,
        index: usize,
        len: usize,
    }
    impl<F> Iterator for Logspace<F>
    where
        F: Float,
    {
        type Item = F;
        #[inline]
        fn next(&mut self) -> Option<F> {
            if self.index >= self.len {
                None
            } else {
                let i = self.index;
                self.index += 1;
                let exponent = self.start + self.step * F::from(i).unwrap();
                Some(self.sign * self.base.powf(exponent))
            }
        }
    }
    impl<F> ExactSizeIterator for Logspace<F> where Logspace<F>: Iterator {}
    #[inline]
    pub fn logspace<F>(base: F, a: F, b: F, n: usize) -> Logspace<F>
    where
        F: Float,
    {
        let step = if n > 1 {
            let num_steps =
                F::from(n - 1).expect("Converting number of steps to `A` must not fail.");
            (b - a) / num_steps
        } else {
            F::zero()
        };
        Logspace {
            sign: base.signum(),
            base: base.abs(),
            start: a,
            step,
            index: 0,
            len: n,
        }
    }
}
mod math_cell {
    use std::cell::Cell;
    use std::cmp::Ordering;
    use std::fmt;
    use std::ops::{Deref, DerefMut};
    #[repr(transparent)]
    #[derive(Default)]
    pub struct MathCell<T>(Cell<T>);
    impl<T> MathCell<T> {}
    impl<T> Deref for MathCell<T> {
        type Target = Cell<T>;
        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T> PartialEq for MathCell<T>
    where
        T: Copy + PartialEq,
    {
        fn eq(&self, rhs: &Self) -> bool {
            self.get() == rhs.get()
        }
    }
    impl<T> Eq for MathCell<T> where T: Copy + Eq {}
    impl<T> PartialOrd for MathCell<T>
    where
        T: Copy + PartialOrd,
    {
        fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
            self.get().partial_cmp(&rhs.get())
        }
    }
}
mod numeric_util {
    use crate::LinalgScalar;
    use std::cmp;
    pub fn unrolled_eq<A, B>(xs: &[A], ys: &[B]) -> bool
    where
        A: PartialEq<B>,
    {
        debug_assert_eq!(xs.len(), ys.len());
        let len = cmp::min(xs.len(), ys.len());
        let mut xs = &xs[..len];
        let mut ys = &ys[..len];
        while xs.len() >= 8 {
            if (xs[0] != ys[0])
                | (xs[1] != ys[1])
                | (xs[2] != ys[2])
                | (xs[3] != ys[3])
                | (xs[4] != ys[4])
                | (xs[5] != ys[5])
                | (xs[6] != ys[6])
                | (xs[7] != ys[7])
            {
                return false;
            }
            xs = &xs[8..];
            ys = &ys[8..];
        }
        for i in 0..xs.len() {
            if xs[i] != ys[i] {
                return false;
            }
        }
        true
    }
}
mod order {
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[non_exhaustive]
    pub enum Order {
        RowMajor,
        ColumnMajor,
    }
}
mod partial {
    use std::ptr;
    #[must_use]
    pub(crate) struct Partial<T> {
        ptr: *mut T,
        pub(crate) len: usize,
    }
    impl<T> Partial<T> {
        pub(crate) unsafe fn new(ptr: *mut T) -> Self {
            Self { ptr, len: 0 }
        }
        pub(crate) fn release_ownership(mut self) -> usize {
            let ret = self.len;
            self.len = 0;
            ret
        }
    }
}
mod shape_builder {
    use crate::dimension::IntoDimension;
    use crate::order::Order;
    use crate::Dimension;
    #[derive(Copy, Clone, Debug)]
    pub struct Shape<D> {
        pub(crate) dim: D,
        pub(crate) strides: Strides<Contiguous>,
    }
    #[derive(Copy, Clone, Debug)]
    pub(crate) enum Contiguous {}
    impl<D> Shape<D> {
        pub(crate) fn is_c(&self) -> bool {
            matches!(self.strides, Strides::C)
        }
    }
    #[derive(Copy, Clone, Debug)]
    pub struct StrideShape<D> {
        pub(crate) dim: D,
        pub(crate) strides: Strides<D>,
    }
    #[derive(Copy, Clone, Debug)]
    pub(crate) enum Strides<D> {
        C,
        F,
        Custom(D),
    }
    impl<D> Strides<D> {
        pub(crate) fn strides_for_dim(self, dim: &D) -> D
        where
            D: Dimension,
        {
            match self {
                Strides::C => dim.default_strides(),
                Strides::F => dim.fortran_strides(),
                Strides::Custom(c) => {
                    debug_assert_eq!(
                        c.ndim(),
                        dim.ndim(),
                        "Custom strides given with {} dimensions, expected {}",
                        c.ndim(),
                        dim.ndim()
                    );
                    c
                }
            }
        }
        pub(crate) fn is_custom(&self) -> bool {
            matches!(*self, Strides::Custom(_))
        }
    }
    pub trait ShapeBuilder {
        type Dim: Dimension;
        type Strides;
        fn into_shape(self) -> Shape<Self::Dim>;
        fn f(self) -> Shape<Self::Dim>;
        fn set_f(self, is_f: bool) -> Shape<Self::Dim>;
        fn strides(self, strides: Self::Strides) -> StrideShape<Self::Dim>;
    }
    impl<T, D> From<T> for StrideShape<D>
    where
        D: Dimension,
        T: ShapeBuilder<Dim = D>,
    {
        fn from(value: T) -> Self {
            let shape = value.into_shape();
            let st = if shape.is_c() { Strides::C } else { Strides::F };
            StrideShape {
                strides: st,
                dim: shape.dim,
            }
        }
    }
    impl<T> ShapeBuilder for T
    where
        T: IntoDimension,
    {
        type Dim = T::Dim;
        type Strides = T;
        fn into_shape(self) -> Shape<Self::Dim> {
            Shape {
                dim: self.into_dimension(),
                strides: Strides::C,
            }
        }
        fn f(self) -> Shape<Self::Dim> {
            self.set_f(true)
        }
        fn set_f(self, is_f: bool) -> Shape<Self::Dim> {
            self.into_shape().set_f(is_f)
        }
        fn strides(self, st: T) -> StrideShape<Self::Dim> {
            self.into_shape().strides(st.into_dimension())
        }
    }
    impl<D> ShapeBuilder for Shape<D>
    where
        D: Dimension,
    {
        type Dim = D;
        type Strides = D;
        fn into_shape(self) -> Shape<D> {
            self
        }
        fn f(self) -> Self {
            self.set_f(true)
        }
        fn set_f(mut self, is_f: bool) -> Self {
            self.strides = if !is_f { Strides::C } else { Strides::F };
            self
        }
        fn strides(self, st: D) -> StrideShape<D> {
            StrideShape {
                dim: self.dim,
                strides: Strides::Custom(st),
            }
        }
    }
    pub trait ShapeArg {
        type Dim: Dimension;
        fn into_shape_and_order(self) -> (Self::Dim, Option<Order>);
    }
}
#[macro_use]
mod slice {
    use crate::dimension::slices_intersect;
    use crate::error::{ErrorKind, ShapeError};
    use crate::{ArrayViewMut, DimAdd, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
    use alloc::vec::Vec;
    use std::convert::TryFrom;
    use std::fmt;
    use std::marker::PhantomData;
    use std::ops::{Deref, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Slice {
        pub start: isize,
        pub end: Option<isize>,
        pub step: isize,
    }
    impl Slice {
        pub fn new(start: isize, end: Option<isize>, step: isize) -> Slice {
            debug_assert_ne!(step, 0, "Slice::new: step must be nonzero");
            Slice { start, end, step }
        }
    }
    #[derive(Clone, Copy, Debug)]
    pub struct NewAxis;
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub enum SliceInfoElem {
        Slice {
            start: isize,
            end: Option<isize>,
            step: isize,
        },
        Index(isize),
        NewAxis,
    }
    copy_and_clone! { SliceInfoElem }
    impl SliceInfoElem {
        pub fn is_index(&self) -> bool {
            matches!(self, SliceInfoElem::Index(_))
        }
        pub fn is_new_axis(&self) -> bool {
            matches!(self, SliceInfoElem::NewAxis)
        }
    }
    macro_rules! impl_slice_variant_from_range {
        ($ self : ty , $ constructor : path , $ index : ty) => {
            impl From<Range<$index>> for $self {
                #[inline]
                fn from(r: Range<$index>) -> $self {
                    $constructor {
                        start: r.start as isize,
                        end: Some(r.end as isize),
                        step: 1,
                    }
                }
            }
            impl From<RangeInclusive<$index>> for $self {
                #[inline]
                fn from(r: RangeInclusive<$index>) -> $self {
                    let end = *r.end() as isize;
                    $constructor {
                        start: *r.start() as isize,
                        end: if end == -1 { None } else { Some(end + 1) },
                        step: 1,
                    }
                }
            }
            impl From<RangeFrom<$index>> for $self {
                #[inline]
                fn from(r: RangeFrom<$index>) -> $self {
                    $constructor {
                        start: r.start as isize,
                        end: None,
                        step: 1,
                    }
                }
            }
            impl From<RangeTo<$index>> for $self {
                #[inline]
                fn from(r: RangeTo<$index>) -> $self {
                    $constructor {
                        start: 0,
                        end: Some(r.end as isize),
                        step: 1,
                    }
                }
            }
            impl From<RangeToInclusive<$index>> for $self {
                #[inline]
                fn from(r: RangeToInclusive<$index>) -> $self {
                    let end = r.end as isize;
                    $constructor {
                        start: 0,
                        end: if end == -1 { None } else { Some(end + 1) },
                        step: 1,
                    }
                }
            }
        };
    }
    impl_slice_variant_from_range!(Slice, Slice, i32);
    macro_rules! impl_sliceinfoelem_from_index {
        ($ index : ty) => {
            impl From<$index> for SliceInfoElem {
                #[inline]
                fn from(r: $index) -> SliceInfoElem {
                    SliceInfoElem::Index(r as isize)
                }
            }
        };
    }
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait SliceArg<D: Dimension>: AsRef<[SliceInfoElem]> {
        type OutDim: Dimension;
        fn in_ndim(&self) -> usize;
        fn out_ndim(&self) -> usize;
        private_decl! {}
    }
    unsafe impl<T, D> SliceArg<D> for &T
    where
        T: SliceArg<D> + ?Sized,
        D: Dimension,
    {
        type OutDim = T::OutDim;
        fn in_ndim(&self) -> usize {
            T::in_ndim(self)
        }
        fn out_ndim(&self) -> usize {
            T::out_ndim(self)
        }
        private_impl! {}
    }
    macro_rules! impl_slicearg_samedim {
        ($ in_dim : ty) => {
            unsafe impl<T, Dout> SliceArg<$in_dim> for SliceInfo<T, $in_dim, Dout>
            where
                T: AsRef<[SliceInfoElem]>,
                Dout: Dimension,
            {
                type OutDim = Dout;
                fn in_ndim(&self) -> usize {
                    self.in_ndim()
                }
                fn out_ndim(&self) -> usize {
                    self.out_ndim()
                }
                private_impl! {}
            }
        };
    }
    unsafe impl SliceArg<IxDyn> for [SliceInfoElem] {
        type OutDim = IxDyn;
        fn in_ndim(&self) -> usize {
            self.iter().filter(|s| !s.is_new_axis()).count()
        }
        fn out_ndim(&self) -> usize {
            self.iter().filter(|s| !s.is_index()).count()
        }
        private_impl! {}
    }
    #[derive(Debug)]
    pub struct SliceInfo<T, Din: Dimension, Dout: Dimension> {
        in_dim: PhantomData<Din>,
        out_dim: PhantomData<Dout>,
        indices: T,
    }
    fn check_dims_for_sliceinfo<Din, Dout>(indices: &[SliceInfoElem]) -> Result<(), ShapeError>
    where
        Din: Dimension,
        Dout: Dimension,
    {
        if let Some(in_ndim) = Din::NDIM {
            if in_ndim != indices.in_ndim() {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
        }
        if let Some(out_ndim) = Dout::NDIM {
            if out_ndim != indices.out_ndim() {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
        }
        Ok(())
    }
    impl<T, Din, Dout> SliceInfo<T, Din, Dout>
    where
        T: AsRef<[SliceInfoElem]>,
        Din: Dimension,
        Dout: Dimension,
    {
    }
    macro_rules! impl_tryfrom_array_for_sliceinfo {
        ($ len : expr) => {
            impl<Din, Dout> TryFrom<[SliceInfoElem; $len]>
                for SliceInfo<[SliceInfoElem; $len], Din, Dout>
            where
                Din: Dimension,
                Dout: Dimension,
            {
                type Error = ShapeError;
                fn try_from(
                    indices: [SliceInfoElem; $len],
                ) -> Result<SliceInfo<[SliceInfoElem; $len], Din, Dout>, ShapeError> {
                    unsafe { Self::new(indices) }
                }
            }
        };
    }
    pub trait SliceNextDim {
        type InDim: Dimension;
        type OutDim: Dimension;
        fn next_in_dim<D>(
            &self,
            _: PhantomData<D>,
        ) -> PhantomData<<D as DimAdd<Self::InDim>>::Output>
        where
            D: Dimension + DimAdd<Self::InDim>,
        {
            PhantomData
        }
        fn next_out_dim<D>(
            &self,
            _: PhantomData<D>,
        ) -> PhantomData<<D as DimAdd<Self::OutDim>>::Output>
        where
            D: Dimension + DimAdd<Self::OutDim>,
        {
            PhantomData
        }
    }
    macro_rules ! impl_slicenextdim { (($ ($ generics : tt) *) , $ self : ty , $ in : ty , $ out : ty) => { impl <$ ($ generics) *> SliceNextDim for $ self { type InDim = $ in ; type OutDim = $ out ; } } ; }
    #[macro_export]
    macro_rules ! s ((@ parse $ in_dim : expr , $ out_dim : expr , [$ ($ stack : tt) *] $ r : expr ;$ s : expr) => { match $ r { r => { let in_dim = $ crate :: SliceNextDim :: next_in_dim (& r , $ in_dim) ; let out_dim = $ crate :: SliceNextDim :: next_out_dim (& r , $ out_dim) ; # [allow (unsafe_code)] unsafe { $ crate :: SliceInfo :: new_unchecked ([$ ($ stack) * $ crate :: s ! (@ convert r , $ s)] , in_dim , out_dim ,) } } } } ; (@ parse $ in_dim : expr , $ out_dim : expr , [$ ($ stack : tt) *] $ r : expr) => { match $ r { r => { let in_dim = $ crate :: SliceNextDim :: next_in_dim (& r , $ in_dim) ; let out_dim = $ crate :: SliceNextDim :: next_out_dim (& r , $ out_dim) ; # [allow (unsafe_code)] unsafe { $ crate :: SliceInfo :: new_unchecked ([$ ($ stack) * $ crate :: s ! (@ convert r)] , in_dim , out_dim ,) } } } } ; (@ parse $ in_dim : expr , $ out_dim : expr , [$ ($ stack : tt) *] $ r : expr ;$ s : expr ,) => { $ crate :: s ! [@ parse $ in_dim , $ out_dim , [$ ($ stack) *] $ r ;$ s] } ; (@ parse $ in_dim : expr , $ out_dim : expr , [$ ($ stack : tt) *] $ r : expr ,) => { $ crate :: s ! [@ parse $ in_dim , $ out_dim , [$ ($ stack) *] $ r] } ; (@ parse $ in_dim : expr , $ out_dim : expr , [$ ($ stack : tt) *] $ r : expr ;$ s : expr , $ ($ t : tt) *) => { match $ r { r => { $ crate :: s ! [@ parse $ crate :: SliceNextDim :: next_in_dim (& r , $ in_dim) , $ crate :: SliceNextDim :: next_out_dim (& r , $ out_dim) , [$ ($ stack) * $ crate :: s ! (@ convert r , $ s) ,] $ ($ t) *] } } } ; (@ parse $ in_dim : expr , $ out_dim : expr , [$ ($ stack : tt) *] $ r : expr , $ ($ t : tt) *) => { match $ r { r => { $ crate :: s ! [@ parse $ crate :: SliceNextDim :: next_in_dim (& r , $ in_dim) , $ crate :: SliceNextDim :: next_out_dim (& r , $ out_dim) , [$ ($ stack) * $ crate :: s ! (@ convert r) ,] $ ($ t) *] } } } ; (@ parse :: core :: marker :: PhantomData ::<$ crate :: Ix0 >, :: core :: marker :: PhantomData ::<$ crate :: Ix0 >, []) => { { # [allow (unsafe_code)] unsafe { $ crate :: SliceInfo :: new_unchecked ([] , :: core :: marker :: PhantomData ::<$ crate :: Ix0 >, :: core :: marker :: PhantomData ::<$ crate :: Ix0 >,) } } } ; (@ parse $ ($ t : tt) *) => { compile_error ! ("Invalid syntax in s![] call.") } ; (@ convert $ r : expr) => { <$ crate :: SliceInfoElem as :: core :: convert :: From < _ >>:: from ($ r) } ; (@ convert $ r : expr , $ s : expr) => { <$ crate :: SliceInfoElem as :: core :: convert :: From < _ >>:: from (<$ crate :: Slice as :: core :: convert :: From < _ >>:: from ($ r) . step_by ($ s as isize)) } ; ($ ($ t : tt) *) => { $ crate :: s ! [@ parse :: core :: marker :: PhantomData ::<$ crate :: Ix0 >, :: core :: marker :: PhantomData ::<$ crate :: Ix0 >, [] $ ($ t) *] } ;) ;
    pub trait MultiSliceArg<'a, A, D>
    where
        A: 'a,
        D: Dimension,
    {
        type Output;
        fn multi_slice_move(&self, view: ArrayViewMut<'a, A, D>) -> Self::Output;
        private_decl! {}
    }
    macro_rules ! impl_multislice_tuple { ([$ ($ but_last : ident) *] $ last : ident) => { impl_multislice_tuple ! (@ def_impl ($ ($ but_last ,) * $ last ,) , [$ ($ but_last) *] $ last) ; } ; (@ def_impl ($ ($ all : ident ,) *) , [$ ($ but_last : ident) *] $ last : ident) => { impl <'a , A , D , $ ($ all ,) *> MultiSliceArg <'a , A , D > for ($ ($ all ,) *) where A : 'a , D : Dimension , $ ($ all : SliceArg < D >,) * { type Output = ($ (ArrayViewMut <'a , A , $ all :: OutDim >,) *) ; fn multi_slice_move (& self , view : ArrayViewMut <'a , A , D >) -> Self :: Output { # [allow (non_snake_case)] let ($ ($ all ,) *) = self ; let shape = view . raw_dim () ; assert ! (! impl_multislice_tuple ! (@ intersects_self & shape , ($ ($ all ,) *))) ; let raw_view = view . into_raw_view_mut () ; unsafe { ($ (raw_view . clone () . slice_move ($ but_last) . deref_into_view_mut () ,) * raw_view . slice_move ($ last) . deref_into_view_mut () ,) } } private_impl ! { } } } ; (@ intersects_self $ shape : expr , ($ head : expr ,)) => { false } ; (@ intersects_self $ shape : expr , ($ head : expr , $ ($ tail : expr ,) *)) => { $ (slices_intersect ($ shape , $ head , $ tail)) ||* || impl_multislice_tuple ! (@ intersects_self $ shape , ($ ($ tail ,) *)) } ; }
}
mod split_at {
    use crate::imp_prelude::*;
    pub(crate) trait SplitAt {
        fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
        where
            Self: Sized;
    }
    pub(crate) trait SplitPreference: SplitAt {
        fn can_split(&self) -> bool;
        fn split_preference(&self) -> (Axis, usize);
        fn split(self) -> (Self, Self)
        where
            Self: Sized,
        {
            let (axis, index) = self.split_preference();
            self.split_at(axis, index)
        }
    }
    impl<D> SplitAt for D
    where
        D: Dimension,
    {
        fn split_at(self, axis: Axis, index: Ix) -> (Self, Self) {
            let mut d1 = self;
            let mut d2 = d1.clone();
            let i = axis.index();
            let len = d1[i];
            d1[i] = index;
            d2[i] = len - index;
            (d1, d2)
        }
    }
}
mod stacking {
    use crate::dimension;
    use crate::error::{from_kind, ErrorKind, ShapeError};
    use crate::imp_prelude::*;
    use alloc::vec::Vec;
    pub fn concatenate<A, D>(
        axis: Axis,
        arrays: &[ArrayView<A, D>],
    ) -> Result<Array<A, D>, ShapeError>
    where
        A: Clone,
        D: RemoveAxis,
    {
        if arrays.is_empty() {
            return Err(from_kind(ErrorKind::Unsupported));
        }
        let mut res_dim = arrays[0].raw_dim();
        if axis.index() >= res_dim.ndim() {
            return Err(from_kind(ErrorKind::OutOfBounds));
        }
        let common_dim = res_dim.remove_axis(axis);
        if arrays
            .iter()
            .any(|a| a.raw_dim().remove_axis(axis) != common_dim)
        {
            return Err(from_kind(ErrorKind::IncompatibleShape));
        }
        let stacked_dim = arrays.iter().fold(0, |acc, a| acc + a.len_of(axis));
        res_dim.set_axis(axis, stacked_dim);
        let new_len = dimension::size_of_shape_checked(&res_dim)?;
        res_dim.set_axis(axis, 0);
        let mut res =
            unsafe { Array::from_shape_vec_unchecked(res_dim, Vec::with_capacity(new_len)) };
        for array in arrays {
            res.append(axis, array.clone())?;
        }
        debug_assert_eq!(res.len_of(axis), stacked_dim);
        Ok(res)
    }
}
mod low_level_util {
    #[must_use]
    pub(crate) struct AbortIfPanic(pub(crate) &'static &'static str);
    impl AbortIfPanic {
        #[inline]
        pub(crate) fn defuse(self) {
            std::mem::forget(self);
        }
    }
}
#[macro_use]
mod zip {
    #[macro_use]
    mod zipmacro {}
    mod ndproducer {
        use crate::imp_prelude::*;
        use crate::Layout;
        use crate::NdIndex;
        #[cfg(not(features = "std"))]
        use alloc::vec::Vec;
        pub trait IntoNdProducer {
            type Item;
            type Dim: Dimension;
            type Output: NdProducer<Dim = Self::Dim, Item = Self::Item>;
            fn into_producer(self) -> Self::Output;
        }
        impl<P> IntoNdProducer for P
        where
            P: NdProducer,
        {
            type Item = P::Item;
            type Dim = P::Dim;
            type Output = Self;
            fn into_producer(self) -> Self::Output {
                self
            }
        }
        pub trait NdProducer {
            type Item;
            type Dim: Dimension;
            type Ptr: Offset<Stride = Self::Stride>;
            type Stride: Copy;
            fn layout(&self) -> Layout;
            fn raw_dim(&self) -> Self::Dim;
            fn equal_dim(&self, dim: &Self::Dim) -> bool {
                self.raw_dim() == *dim
            }
            fn as_ptr(&self) -> Self::Ptr;
            unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item;
            unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr;
            fn stride_of(&self, axis: Axis) -> <Self::Ptr as Offset>::Stride;
            fn contiguous_stride(&self) -> Self::Stride;
            fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
            where
                Self: Sized;
            private_decl! {}
        }
        pub trait Offset: Copy {
            type Stride: Copy;
            unsafe fn stride_offset(self, s: Self::Stride, index: usize) -> Self;
            private_decl! {}
        }
        impl<T> Offset for *const T {
            type Stride = isize;
            unsafe fn stride_offset(self, s: Self::Stride, index: usize) -> Self {
                self.offset(s * (index as isize))
            }
            private_impl! {}
        }
        impl<T> Offset for *mut T {
            type Stride = isize;
            unsafe fn stride_offset(self, s: Self::Stride, index: usize) -> Self {
                self.offset(s * (index as isize))
            }
            private_impl! {}
        }
        impl<'a, A: 'a, S, D> IntoNdProducer for &'a ArrayBase<S, D>
        where
            D: Dimension,
            S: Data<Elem = A>,
        {
            type Item = &'a A;
            type Dim = D;
            type Output = ArrayView<'a, A, D>;
            fn into_producer(self) -> Self::Output {
                self.view()
            }
        }
        impl<'a, A: 'a, S, D> IntoNdProducer for &'a mut ArrayBase<S, D>
        where
            D: Dimension,
            S: DataMut<Elem = A>,
        {
            type Item = &'a mut A;
            type Dim = D;
            type Output = ArrayViewMut<'a, A, D>;
            fn into_producer(self) -> Self::Output {
                self.view_mut()
            }
        }
        impl<'a, A, D: Dimension> NdProducer for ArrayView<'a, A, D> {
            type Item = &'a A;
            type Dim = D;
            type Ptr = *mut A;
            type Stride = isize;
            private_impl! {}
            fn raw_dim(&self) -> Self::Dim {
                self.raw_dim()
            }
            fn as_ptr(&self) -> *mut A {
                self.as_ptr() as _
            }
            fn layout(&self) -> Layout {
                self.layout_impl()
            }
            unsafe fn as_ref(&self, ptr: *mut A) -> Self::Item {
                &*ptr
            }
            unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
                self.ptr.as_ptr().offset(i.index_unchecked(&self.strides))
            }
            fn stride_of(&self, axis: Axis) -> isize {
                self.stride_of(axis)
            }
            #[inline(always)]
            fn contiguous_stride(&self) -> Self::Stride {
                1
            }
            fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
                self.split_at(axis, index)
            }
        }
        impl<'a, A, D: Dimension> NdProducer for ArrayViewMut<'a, A, D> {
            type Item = &'a mut A;
            type Dim = D;
            type Ptr = *mut A;
            type Stride = isize;
            private_impl! {}
            fn raw_dim(&self) -> Self::Dim {
                self.raw_dim()
            }
            fn as_ptr(&self) -> *mut A {
                self.as_ptr() as _
            }
            fn layout(&self) -> Layout {
                self.layout_impl()
            }
            unsafe fn as_ref(&self, ptr: *mut A) -> Self::Item {
                &mut *ptr
            }
            unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
                self.ptr.as_ptr().offset(i.index_unchecked(&self.strides))
            }
            fn stride_of(&self, axis: Axis) -> isize {
                self.stride_of(axis)
            }
            #[inline(always)]
            fn contiguous_stride(&self) -> Self::Stride {
                1
            }
            fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
                self.split_at(axis, index)
            }
        }
        impl<A, D: Dimension> NdProducer for RawArrayView<A, D> {
            type Item = *const A;
            type Dim = D;
            type Ptr = *const A;
            type Stride = isize;
            private_impl! {}
            fn raw_dim(&self) -> Self::Dim {
                self.raw_dim()
            }
            fn as_ptr(&self) -> *const A {
                self.as_ptr()
            }
            fn layout(&self) -> Layout {
                self.layout_impl()
            }
            unsafe fn as_ref(&self, ptr: *const A) -> *const A {
                ptr
            }
            unsafe fn uget_ptr(&self, i: &Self::Dim) -> *const A {
                self.ptr.as_ptr().offset(i.index_unchecked(&self.strides))
            }
            fn stride_of(&self, axis: Axis) -> isize {
                self.stride_of(axis)
            }
            #[inline(always)]
            fn contiguous_stride(&self) -> Self::Stride {
                1
            }
            fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
                self.split_at(axis, index)
            }
        }
        impl<A, D: Dimension> NdProducer for RawArrayViewMut<A, D> {
            type Item = *mut A;
            type Dim = D;
            type Ptr = *mut A;
            type Stride = isize;
            private_impl! {}
            fn raw_dim(&self) -> Self::Dim {
                self.raw_dim()
            }
            fn as_ptr(&self) -> *mut A {
                self.as_ptr() as _
            }
            fn layout(&self) -> Layout {
                self.layout_impl()
            }
            unsafe fn as_ref(&self, ptr: *mut A) -> *mut A {
                ptr
            }
            unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
                self.ptr.as_ptr().offset(i.index_unchecked(&self.strides))
            }
            fn stride_of(&self, axis: Axis) -> isize {
                self.stride_of(axis)
            }
            #[inline(always)]
            fn contiguous_stride(&self) -> Self::Stride {
                1
            }
            fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
                self.split_at(axis, index)
            }
        }
    }
    pub use self::ndproducer::{IntoNdProducer, NdProducer, Offset};
    use crate::dimension;
    use crate::imp_prelude::*;
    use crate::indexes::{indices, Indices};
    use crate::partial::Partial;
    use crate::split_at::{SplitAt, SplitPreference};
    use crate::AssignElem;
    use crate::IntoDimension;
    use crate::Layout;
    macro_rules! fold_while {
        ($ e : expr) => {
            match $e {
                FoldWhile::Continue(x) => x,
                x => return x,
            }
        };
    }
    trait Broadcast<E>
    where
        E: IntoDimension,
    {
        type Output: NdProducer<Dim = E::Dim>;
        fn broadcast_unwrap(self, shape: E) -> Self::Output;
        private_decl! {}
    }
    fn array_layout<D: Dimension>(dim: &D, strides: &D) -> Layout {
        let n = dim.ndim();
        if dimension::is_layout_c(dim, strides) {
            if n <= 1 || dim.slice().iter().filter(|&&len| len > 1).count() <= 1 {
                Layout::one_dimensional()
            } else {
                Layout::c()
            }
        } else if n > 1 && dimension::is_layout_f(dim, strides) {
            Layout::f()
        } else if n > 1 {
            if dim[0] > 1 && strides[0] == 1 {
                Layout::fpref()
            } else if dim[n - 1] > 1 && strides[n - 1] == 1 {
                Layout::cpref()
            } else {
                Layout::none()
            }
        } else {
            Layout::none()
        }
    }
    impl<S, D> ArrayBase<S, D>
    where
        S: RawData,
        D: Dimension,
    {
        pub(crate) fn layout_impl(&self) -> Layout {
            array_layout(&self.dim, &self.strides)
        }
    }
    impl<'a, A, D, E> Broadcast<E> for ArrayView<'a, A, D>
    where
        E: IntoDimension,
        D: Dimension,
    {
        type Output = ArrayView<'a, A, E::Dim>;
        fn broadcast_unwrap(self, shape: E) -> Self::Output {
            let res: ArrayView<'_, A, E::Dim> = (&self).broadcast_unwrap(shape.into_dimension());
            unsafe { ArrayView::new(res.ptr, res.dim, res.strides) }
        }
        private_impl! {}
    }
    trait ZippableTuple: Sized {
        type Item;
        type Ptr: OffsetTuple<Args = Self::Stride> + Copy;
        type Dim: Dimension;
        type Stride: Copy;
        fn as_ptr(&self) -> Self::Ptr;
        unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item;
        unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr;
        fn stride_of(&self, index: usize) -> Self::Stride;
        fn contiguous_stride(&self) -> Self::Stride;
        fn split_at(self, axis: Axis, index: usize) -> (Self, Self);
    }
    #[doc = " Lock step function application across several arrays or other producers."]
    #[doc = ""]
    #[doc = " Zip allows matching several producers to each other elementwise and applying"]
    #[doc = " a function over all tuples of elements (one item from each input at"]
    #[doc = " a time)."]
    #[doc = ""]
    #[doc = " In general, the zip uses a tuple of producers"]
    #[doc = " ([`NdProducer`] trait) that all have to be of the"]
    #[doc = " same shape. The NdProducer implementation defines what its item type is"]
    #[doc = " (for example if it's a shared reference, mutable reference or an array"]
    #[doc = " view etc)."]
    #[doc = ""]
    #[doc = " If all the input arrays are of the same memory layout the zip performs much"]
    #[doc = " better and the compiler can usually vectorize the loop (if applicable)."]
    #[doc = ""]
    #[doc = " The order elements are visited is not specified. The producers don’t have to"]
    #[doc = " have the same item type."]
    #[doc = ""]
    #[doc = " The `Zip` has two methods for function application: `for_each` and"]
    #[doc = " `fold_while`. The zip object can be split, which allows parallelization."]
    #[doc = " A read-only zip object (no mutable producers) can be cloned."]
    #[doc = ""]
    #[doc = " See also the [`azip!()`] which offers a convenient shorthand"]
    #[doc = " to common ways to use `Zip`."]
    #[doc = ""]
    #[doc = " ```"]
    #[doc = " use ndarray::Zip;"]
    #[doc = " use ndarray::Array2;"]
    #[doc = ""]
    #[doc = " type M = Array2<f64>;"]
    #[doc = ""]
    #[doc = " // Create four 2d arrays of the same size"]
    #[doc = " let mut a = M::zeros((64, 32));"]
    #[doc = " let b = M::from_elem(a.dim(), 1.);"]
    #[doc = " let c = M::from_elem(a.dim(), 2.);"]
    #[doc = " let d = M::from_elem(a.dim(), 3.);"]
    #[doc = ""]
    #[doc = " // Example 1: Perform an elementwise arithmetic operation across"]
    #[doc = " // the four arrays a, b, c, d."]
    #[doc = ""]
    #[doc = " Zip::from(&mut a)"]
    #[doc = "     .and(&b)"]
    #[doc = "     .and(&c)"]
    #[doc = "     .and(&d)"]
    #[doc = "     .for_each(|w, &x, &y, &z| {"]
    #[doc = "         *w += x + y * z;"]
    #[doc = "     });"]
    #[doc = ""]
    #[doc = " // Example 2: Create a new array `totals` with one entry per row of `a`."]
    #[doc = " //  Use Zip to traverse the rows of `a` and assign to the corresponding"]
    #[doc = " //  entry in `totals` with the sum across each row."]
    #[doc = " //  This is possible because the producer for `totals` and the row producer"]
    #[doc = " //  for `a` have the same shape and dimensionality."]
    #[doc = " //  The rows producer yields one array view (`row`) per iteration."]
    #[doc = ""]
    #[doc = " use ndarray::{Array1, Axis};"]
    #[doc = ""]
    #[doc = " let mut totals = Array1::zeros(a.nrows());"]
    #[doc = ""]
    #[doc = " Zip::from(&mut totals)"]
    #[doc = "     .and(a.rows())"]
    #[doc = "     .for_each(|totals, row| *totals = row.sum());"]
    #[doc = ""]
    #[doc = " // Check the result against the built in `.sum_axis()` along axis 1."]
    #[doc = " assert_eq!(totals, a.sum_axis(Axis(1)));"]
    #[doc = ""]
    #[doc = ""]
    #[doc = " // Example 3: Recreate Example 2 using map_collect to make a new array"]
    #[doc = ""]
    #[doc = " let totals2 = Zip::from(a.rows()).map_collect(|row| row.sum());"]
    #[doc = ""]
    #[doc = " // Check the result against the previous example."]
    #[doc = " assert_eq!(totals, totals2);"]
    #[doc = " ```"]
    #[derive(Debug, Clone)]
    #[must_use = "zipping producers is lazy and does nothing unless consumed"]
    pub struct Zip<Parts, D> {
        parts: Parts,
        dimension: D,
        layout: Layout,
        layout_tendency: i32,
    }
    impl<P, D> Zip<(P,), D>
    where
        D: Dimension,
        P: NdProducer<Dim = D>,
    {
        pub fn from<IP>(p: IP) -> Self
        where
            IP: IntoNdProducer<Dim = D, Output = P, Item = P::Item>,
        {
            let array = p.into_producer();
            let dim = array.raw_dim();
            let layout = array.layout();
            Zip {
                dimension: dim,
                layout,
                parts: (array,),
                layout_tendency: layout.tendency(),
            }
        }
    }
    #[inline]
    fn zip_dimension_check<D, P>(dimension: &D, part: &P)
    where
        D: Dimension,
        P: NdProducer<Dim = D>,
    {
        ndassert!(
            part.equal_dim(dimension),
            "Zip: Producer dimension mismatch, expected: {:?}, got: {:?}",
            dimension,
            part.raw_dim()
        );
    }
    impl<Parts, D> Zip<Parts, D>
    where
        D: Dimension,
    {
        pub fn size(&self) -> usize {
            self.dimension.size()
        }
        fn len_of(&self, axis: Axis) -> usize {
            self.dimension[axis.index()]
        }
        fn prefer_f(&self) -> bool {
            !self.layout.is(Layout::CORDER)
                && (self.layout.is(Layout::FORDER) || self.layout_tendency < 0)
        }
        fn max_stride_axis(&self) -> Axis {
            let i = if self.prefer_f() {
                self.dimension
                    .slice()
                    .iter()
                    .rposition(|&len| len > 1)
                    .unwrap_or(self.dimension.ndim() - 1)
            } else {
                self.dimension
                    .slice()
                    .iter()
                    .position(|&len| len > 1)
                    .unwrap_or(0)
            };
            Axis(i)
        }
    }
    impl<P, D> Zip<P, D>
    where
        D: Dimension,
    {
        fn for_each_core<F, Acc>(&mut self, acc: Acc, mut function: F) -> FoldWhile<Acc>
        where
            F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
            P: ZippableTuple<Dim = D>,
        {
            if self.dimension.ndim() == 0 {
                function(acc, unsafe { self.parts.as_ref(self.parts.as_ptr()) })
            } else if self.layout.is(Layout::CORDER | Layout::FORDER) {
                self.for_each_core_contiguous(acc, function)
            } else {
                self.for_each_core_strided(acc, function)
            }
        }
        fn for_each_core_contiguous<F, Acc>(&mut self, acc: Acc, mut function: F) -> FoldWhile<Acc>
        where
            F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
            P: ZippableTuple<Dim = D>,
        {
            debug_assert!(self.layout.is(Layout::CORDER | Layout::FORDER));
            let size = self.dimension.size();
            let ptrs = self.parts.as_ptr();
            let inner_strides = self.parts.contiguous_stride();
            unsafe { self.inner(acc, ptrs, inner_strides, size, &mut function) }
        }
        unsafe fn inner<F, Acc>(
            &self,
            mut acc: Acc,
            ptr: P::Ptr,
            strides: P::Stride,
            len: usize,
            function: &mut F,
        ) -> FoldWhile<Acc>
        where
            F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
            P: ZippableTuple,
        {
            let mut i = 0;
            while i < len {
                let p = ptr.stride_offset(strides, i);
                acc = fold_while!(function(acc, self.parts.as_ref(p)));
                i += 1;
            }
            FoldWhile::Continue(acc)
        }
        fn for_each_core_strided<F, Acc>(&mut self, acc: Acc, function: F) -> FoldWhile<Acc>
        where
            F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
            P: ZippableTuple<Dim = D>,
        {
            let n = self.dimension.ndim();
            if n == 0 {
                panic!("Unreachable: ndim == 0 is contiguous")
            }
            if n == 1 || self.layout_tendency >= 0 {
                self.for_each_core_strided_c(acc, function)
            } else {
                self.for_each_core_strided_f(acc, function)
            }
        }
        fn for_each_core_strided_c<F, Acc>(
            &mut self,
            mut acc: Acc,
            mut function: F,
        ) -> FoldWhile<Acc>
        where
            F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
            P: ZippableTuple<Dim = D>,
        {
            let n = self.dimension.ndim();
            let unroll_axis = n - 1;
            let inner_len = self.dimension[unroll_axis];
            self.dimension[unroll_axis] = 1;
            let mut index_ = self.dimension.first_index();
            let inner_strides = self.parts.stride_of(unroll_axis);
            while let Some(index) = index_ {
                unsafe {
                    let ptr = self.parts.uget_ptr(&index);
                    acc =
                        fold_while![self.inner(acc, ptr, inner_strides, inner_len, &mut function)];
                }
                index_ = self.dimension.next_for(index);
            }
            FoldWhile::Continue(acc)
        }
        fn for_each_core_strided_f<F, Acc>(
            &mut self,
            mut acc: Acc,
            mut function: F,
        ) -> FoldWhile<Acc>
        where
            F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
            P: ZippableTuple<Dim = D>,
        {
            let unroll_axis = 0;
            let inner_len = self.dimension[unroll_axis];
            self.dimension[unroll_axis] = 1;
            let index_ = self.dimension.first_index();
            let inner_strides = self.parts.stride_of(unroll_axis);
            if let Some(mut index) = index_ {
                loop {
                    unsafe {
                        let ptr = self.parts.uget_ptr(&index);
                        acc = fold_while![self.inner(
                            acc,
                            ptr,
                            inner_strides,
                            inner_len,
                            &mut function
                        )];
                    }
                    if !self.dimension.next_for_f(&mut index) {
                        break;
                    }
                }
            }
            FoldWhile::Continue(acc)
        }
    }
    impl<D, P1, P2> Zip<(P1, P2), D>
    where
        D: Dimension,
        P1: NdProducer<Dim = D>,
        P1: NdProducer<Dim = D>,
    {
        #[inline]
        pub(crate) fn debug_assert_c_order(self) -> Self {
            debug_assert!(
                self.layout.is(Layout::CORDER)
                    || self.layout_tendency >= 0
                    || self.dimension.slice().iter().filter(|&&d| d > 1).count() <= 1,
                "Assertion failed: traversal is not c-order or 1D for \
                      layout {:?}, tendency {}, dimension {:?}",
                self.layout,
                self.layout_tendency,
                self.dimension
            );
            self
        }
    }
    trait OffsetTuple {
        type Args;
        unsafe fn stride_offset(self, stride: Self::Args, index: usize) -> Self;
    }
    macro_rules ! offset_impl { ($ ([$ ($ param : ident) *] [$ ($ q : ident) *] ,) +) => { $ (# [allow (non_snake_case)] impl <$ ($ param : Offset) ,*> OffsetTuple for ($ ($ param ,) *) { type Args = ($ ($ param :: Stride ,) *) ; unsafe fn stride_offset (self , stride : Self :: Args , index : usize) -> Self { let ($ ($ param ,) *) = self ; let ($ ($ q ,) *) = stride ; ($ (Offset :: stride_offset ($ param , $ q , index) ,) *) } }) + } }
    offset_impl! { [A] [a] , [A B] [a b] , [A B C] [a b c] , [A B C D] [a b c d] , [A B C D E] [a b c d e] , [A B C D E F] [a b c d e f] , }
    macro_rules ! zipt_impl { ($ ([$ ($ p : ident) *] [$ ($ q : ident) *] ,) +) => { $ (# [allow (non_snake_case)] impl < Dim : Dimension , $ ($ p : NdProducer < Dim = Dim >) ,*> ZippableTuple for ($ ($ p ,) *) { type Item = ($ ($ p :: Item ,) *) ; type Ptr = ($ ($ p :: Ptr ,) *) ; type Dim = Dim ; type Stride = ($ ($ p :: Stride ,) *) ; fn stride_of (& self , index : usize) -> Self :: Stride { let ($ (ref $ p ,) *) = * self ; ($ ($ p . stride_of (Axis (index)) ,) *) } fn contiguous_stride (& self) -> Self :: Stride { let ($ (ref $ p ,) *) = * self ; ($ ($ p . contiguous_stride () ,) *) } fn as_ptr (& self) -> Self :: Ptr { let ($ (ref $ p ,) *) = * self ; ($ ($ p . as_ptr () ,) *) } unsafe fn as_ref (& self , ptr : Self :: Ptr) -> Self :: Item { let ($ (ref $ q ,) *) = * self ; let ($ ($ p ,) *) = ptr ; ($ ($ q . as_ref ($ p) ,) *) } unsafe fn uget_ptr (& self , i : & Self :: Dim) -> Self :: Ptr { let ($ (ref $ p ,) *) = * self ; ($ ($ p . uget_ptr (i) ,) *) } fn split_at (self , axis : Axis , index : Ix) -> (Self , Self) { let ($ ($ p ,) *) = self ; let ($ ($ p ,) *) = ($ ($ p . split_at (axis , index) ,) *) ; (($ ($ p . 0 ,) *) , ($ ($ p . 1 ,) *)) } }) + } }
    zipt_impl! { [A] [a] , [A B] [a b] , [A B C] [a b c] , [A B C D] [a b c d] , [A B C D E] [a b c d e] , [A B C D E F] [a b c d e f] , }
    macro_rules ! map_impl { ($ ([$ notlast : ident $ ($ p : ident) *] ,) +) => { $ (# [allow (non_snake_case)] impl < D , $ ($ p) ,*> Zip < ($ ($ p ,) *) , D > where D : Dimension , $ ($ p : NdProducer < Dim = D > ,) * { # [doc = " Apply a function to all elements of the input arrays,"] # [doc = " visiting elements in lock step."] pub fn for_each < F > (mut self , mut function : F) where F : FnMut ($ ($ p :: Item) ,*) { self . for_each_core (() , move | () , args | { let ($ ($ p ,) *) = args ; FoldWhile :: Continue (function ($ ($ p) ,*)) }) ; } # [doc = " Apply a function to all elements of the input arrays,"] # [doc = " visiting elements in lock step."] # [deprecated (note = "Renamed to .for_each()" , since = "0.15.0")] pub fn apply < F > (self , function : F) where F : FnMut ($ ($ p :: Item) ,*) { self . for_each (function) } # [doc = " Apply a fold function to all elements of the input arrays,"] # [doc = " visiting elements in lock step."] # [doc = ""] # [doc = " # Example"] # [doc = ""] # [doc = " The expression `tr(AᵀB)` can be more efficiently computed as"] # [doc = " the equivalent expression `∑ᵢⱼ(A∘B)ᵢⱼ` (i.e. the sum of the"] # [doc = " elements of the entry-wise product). It would be possible to"] # [doc = " evaluate this expression by first computing the entry-wise"] # [doc = " product, `A∘B`, and then computing the elementwise sum of that"] # [doc = " product, but it's possible to do this in a single loop (and"] # [doc = " avoid an extra heap allocation if `A` and `B` can't be"] # [doc = " consumed) by using `Zip`:"] # [doc = ""] # [doc = " ```"] # [doc = " use ndarray::{array, Zip};"] # [doc = ""] # [doc = " let a = array![[1, 5], [3, 7]];"] # [doc = " let b = array![[2, 4], [8, 6]];"] # [doc = ""] # [doc = " // Without using `Zip`. This involves two loops and an extra"] # [doc = " // heap allocation for the result of `&a * &b`."] # [doc = " let sum_prod_nonzip = (&a * &b).sum();"] # [doc = " // Using `Zip`. This is a single loop without any heap allocations."] # [doc = " let sum_prod_zip = Zip::from(&a).and(&b).fold(0, |acc, a, b| acc + a * b);"] # [doc = ""] # [doc = " assert_eq!(sum_prod_nonzip, sum_prod_zip);"] # [doc = " ```"] pub fn fold < F , Acc > (mut self , acc : Acc , mut function : F) -> Acc where F : FnMut (Acc , $ ($ p :: Item) ,*) -> Acc , { self . for_each_core (acc , move | acc , args | { let ($ ($ p ,) *) = args ; FoldWhile :: Continue (function (acc , $ ($ p) ,*)) }) . into_inner () } # [doc = " Apply a fold function to the input arrays while the return"] # [doc = " value is `FoldWhile::Continue`, visiting elements in lock step."] # [doc = ""] pub fn fold_while < F , Acc > (mut self , acc : Acc , mut function : F) -> FoldWhile < Acc > where F : FnMut (Acc , $ ($ p :: Item) ,*) -> FoldWhile < Acc > { self . for_each_core (acc , move | acc , args | { let ($ ($ p ,) *) = args ; function (acc , $ ($ p) ,*) }) } # [doc = " Tests if every element of the iterator matches a predicate."] # [doc = ""] # [doc = " Returns `true` if `predicate` evaluates to `true` for all elements."] # [doc = " Returns `true` if the input arrays are empty."] # [doc = ""] # [doc = " Example:"] # [doc = ""] # [doc = " ```"] # [doc = " use ndarray::{array, Zip};"] # [doc = " let a = array![1, 2, 3];"] # [doc = " let b = array![1, 4, 9];"] # [doc = " assert!(Zip::from(&a).and(&b).all(|&a, &b| a * a == b));"] # [doc = " ```"] pub fn all < F > (mut self , mut predicate : F) -> bool where F : FnMut ($ ($ p :: Item) ,*) -> bool { ! self . for_each_core (() , move | _ , args | { let ($ ($ p ,) *) = args ; if predicate ($ ($ p) ,*) { FoldWhile :: Continue (()) } else { FoldWhile :: Done (()) } }) . is_done () } expand_if ! (@ bool [$ notlast] # [doc = " Include the producer `p` in the Zip."] # [doc = ""] # [doc = " ***Panics*** if `p`’s shape doesn’t match the Zip’s exactly."] pub fn and < P > (self , p : P) -> Zip < ($ ($ p ,) * P :: Output ,) , D > where P : IntoNdProducer < Dim = D >, { let part = p . into_producer () ; zip_dimension_check (& self . dimension , & part) ; self . build_and (part) } # [doc = " Include the producer `p` in the Zip."] # [doc = ""] # [doc = " ## Safety"] # [doc = ""] # [doc = " The caller must ensure that the producer's shape is equal to the Zip's shape."] # [doc = " Uses assertions when debug assertions are enabled."] # [allow (unused)] pub (crate) unsafe fn and_unchecked < P > (self , p : P) -> Zip < ($ ($ p ,) * P :: Output ,) , D > where P : IntoNdProducer < Dim = D >, { # [cfg (debug_assertions)] { self . and (p) } # [cfg (not (debug_assertions))] { self . build_and (p . into_producer ()) } } # [doc = " Include the producer `p` in the Zip, broadcasting if needed."] # [doc = ""] # [doc = " If their shapes disagree, `rhs` is broadcast to the shape of `self`."] # [doc = ""] # [doc = " ***Panics*** if broadcasting isn’t possible."] pub fn and_broadcast <'a , P , D2 , Elem > (self , p : P) -> Zip < ($ ($ p ,) * ArrayView <'a , Elem , D >,) , D > where P : IntoNdProducer < Dim = D2 , Output = ArrayView <'a , Elem , D2 >, Item =&'a Elem >, D2 : Dimension , { let part = p . into_producer () . broadcast_unwrap (self . dimension . clone ()) ; self . build_and (part) } fn build_and < P > (self , part : P) -> Zip < ($ ($ p ,) * P ,) , D > where P : NdProducer < Dim = D >, { let part_layout = part . layout () ; let ($ ($ p ,) *) = self . parts ; Zip { parts : ($ ($ p ,) * part ,) , layout : self . layout . intersect (part_layout) , dimension : self . dimension , layout_tendency : self . layout_tendency + part_layout . tendency () , } } # [doc = " Map and collect the results into a new array, which has the same size as the"] # [doc = " inputs."] # [doc = ""] # [doc = " If all inputs are c- or f-order respectively, that is preserved in the output."] pub fn map_collect < R > (self , f : impl FnMut ($ ($ p :: Item ,) *) -> R) -> Array < R , D > { self . map_collect_owned (f) } pub (crate) fn map_collect_owned < S , R > (self , f : impl FnMut ($ ($ p :: Item ,) *) -> R) -> ArrayBase < S , D > where S : DataOwned < Elem = R > { let shape = self . dimension . clone () . set_f (self . prefer_f ()) ; let output = < ArrayBase < S , D >>:: build_uninit (shape , | output | { unsafe { let output_view = output . into_raw_view_mut () . cast ::< R > () ; self . and (output_view) . collect_with_partial (f) . release_ownership () ; } }) ; unsafe { output . assume_init () } } # [doc = " Map and collect the results into a new array, which has the same size as the"] # [doc = " inputs."] # [doc = ""] # [doc = " If all inputs are c- or f-order respectively, that is preserved in the output."] # [deprecated (note = "Renamed to .map_collect()" , since = "0.15.0")] pub fn apply_collect < R > (self , f : impl FnMut ($ ($ p :: Item ,) *) -> R) -> Array < R , D > { self . map_collect (f) } # [doc = " Map and assign the results into the producer `into`, which should have the same"] # [doc = " size as the other inputs."] # [doc = ""] # [doc = " The producer should have assignable items as dictated by the `AssignElem` trait,"] # [doc = " for example `&mut R`."] pub fn map_assign_into < R , Q > (self , into : Q , mut f : impl FnMut ($ ($ p :: Item ,) *) -> R) where Q : IntoNdProducer < Dim = D >, Q :: Item : AssignElem < R > { self . and (into) . for_each (move |$ ($ p ,) * output_ | { output_ . assign_elem (f ($ ($ p) ,*)) ; }) ; } # [doc = " Map and assign the results into the producer `into`, which should have the same"] # [doc = " size as the other inputs."] # [doc = ""] # [doc = " The producer should have assignable items as dictated by the `AssignElem` trait,"] # [doc = " for example `&mut R`."] # [deprecated (note = "Renamed to .map_assign_into()" , since = "0.15.0")] pub fn apply_assign_into < R , Q > (self , into : Q , f : impl FnMut ($ ($ p :: Item ,) *) -> R) where Q : IntoNdProducer < Dim = D >, Q :: Item : AssignElem < R > { self . map_assign_into (into , f) }) ; # [doc = " Split the `Zip` evenly in two."] # [doc = ""] # [doc = " It will be split in the way that best preserves element locality."] pub fn split (self) -> (Self , Self) { debug_assert_ne ! (self . size () , 0 , "Attempt to split empty zip") ; debug_assert_ne ! (self . size () , 1 , "Attempt to split zip with 1 elem") ; SplitPreference :: split (self) } } expand_if ! (@ bool [$ notlast] # [allow (non_snake_case)] impl < D , PLast , R , $ ($ p) ,*> Zip < ($ ($ p ,) * PLast) , D > where D : Dimension , $ ($ p : NdProducer < Dim = D > ,) * PLast : NdProducer < Dim = D , Item = * mut R , Ptr = * mut R , Stride = isize >, { # [doc = " The inner workings of map_collect and par_map_collect"] # [doc = ""] # [doc = " Apply the function and collect the results into the output (last producer)"] # [doc = " which should be a raw array view; a Partial that owns the written"] # [doc = " elements is returned."] # [doc = ""] # [doc = " Elements will be overwritten in place (in the sense of std::ptr::write)."] # [doc = ""] # [doc = " ## Safety"] # [doc = ""] # [doc = " The last producer is a RawArrayViewMut and must be safe to write into."] # [doc = " The producer must be c- or f-contig and have the same layout tendency"] # [doc = " as the whole Zip."] # [doc = ""] # [doc = " The returned Partial's proxy ownership of the elements must be handled,"] # [doc = " before the array the raw view points to realizes its ownership."] pub (crate) unsafe fn collect_with_partial < F > (self , mut f : F) -> Partial < R > where F : FnMut ($ ($ p :: Item ,) *) -> R { let (.., ref output) = & self . parts ; if cfg ! (debug_assertions) { let out_layout = output . layout () ; assert ! (out_layout . is (Layout :: CORDER | Layout :: FORDER)) ; assert ! ((self . layout_tendency <= 0 && out_layout . tendency () <= 0) || (self . layout_tendency >= 0 && out_layout . tendency () >= 0) , "layout tendency violation for self layout {:?}, output layout {:?},\
                            output shape {:?}" , self . layout , out_layout , output . raw_dim ()) ; } let mut partial = Partial :: new (output . as_ptr ()) ; let partial_len = & mut partial . len ; self . for_each (move |$ ($ p ,) * output_elem : * mut R | { output_elem . write (f ($ ($ p) ,*)) ; if std :: mem :: needs_drop ::< R > () { * partial_len += 1 ; } }) ; partial } }) ; impl < D , $ ($ p) ,*> SplitPreference for Zip < ($ ($ p ,) *) , D > where D : Dimension , $ ($ p : NdProducer < Dim = D > ,) * { fn can_split (& self) -> bool { self . size () > 1 } fn split_preference (& self) -> (Axis , usize) { let axis = self . max_stride_axis () ; let index = self . len_of (axis) / 2 ; (axis , index) } } impl < D , $ ($ p) ,*> SplitAt for Zip < ($ ($ p ,) *) , D > where D : Dimension , $ ($ p : NdProducer < Dim = D > ,) * { fn split_at (self , axis : Axis , index : usize) -> (Self , Self) { let (p1 , p2) = self . parts . split_at (axis , index) ; let (d1 , d2) = self . dimension . split_at (axis , index) ; (Zip { dimension : d1 , layout : self . layout , parts : p1 , layout_tendency : self . layout_tendency , } , Zip { dimension : d2 , layout : self . layout , parts : p2 , layout_tendency : self . layout_tendency , }) } }) + } }
    map_impl! { [true P1] , [true P1 P2] , [true P1 P2 P3] , [true P1 P2 P3 P4] , [true P1 P2 P3 P4 P5] , [false P1 P2 P3 P4 P5 P6] , }
    #[derive(Debug, Copy, Clone)]
    pub enum FoldWhile<T> {
        Continue(T),
        Done(T),
    }
    impl<T> FoldWhile<T> {
        pub fn into_inner(self) -> T {
            match self {
                FoldWhile::Continue(x) | FoldWhile::Done(x) => x,
            }
        }
        pub fn is_done(&self) -> bool {
            match *self {
                FoldWhile::Continue(_) => false,
                FoldWhile::Done(_) => true,
            }
        }
    }
}
mod dimension {
    pub(crate) use self::axes::axes_of;
    pub use self::axes::{Axes, AxisDescription};
    pub use self::axis::Axis;
    pub use self::broadcast::DimMax;
    pub use self::conversion::IntoDimension;
    pub use self::dim::*;
    pub use self::dimension_trait::Dimension;
    pub use self::dynindeximpl::IxDynImpl;
    pub use self::ndindex::NdIndex;
    pub use self::ops::DimAdd;
    pub use self::remove_axis::RemoveAxis;
    pub(crate) use self::reshape::reshape_dim;
    use crate::error::{from_kind, ErrorKind, ShapeError};
    use crate::shape_builder::Strides;
    use crate::slice::SliceArg;
    use crate::{Ix, Ixs, Slice, SliceInfoElem};
    use num_integer::div_floor;
    use std::mem;
    #[macro_use]
    mod macros {
        macro_rules! get {
            ($ dim : expr , $ i : expr) => {
                (*$dim.ix())[$i]
            };
        }
        macro_rules! getm {
            ($ dim : expr , $ i : expr) => {
                (*$dim.ixm())[$i]
            };
        }
    }
    mod axes {
        use crate::{Axis, Dimension, Ix, Ixs};
        pub(crate) fn axes_of<'a, D>(d: &'a D, strides: &'a D) -> Axes<'a, D>
        where
            D: Dimension,
        {
            Axes {
                dim: d,
                strides,
                start: 0,
                end: d.ndim(),
            }
        }
        #[derive(Debug)]
        pub struct Axes<'a, D> {
            dim: &'a D,
            strides: &'a D,
            start: usize,
            end: usize,
        }
        #[derive(Debug)]
        pub struct AxisDescription {
            pub axis: Axis,
            pub len: usize,
            pub stride: isize,
        }
        impl<'a, D> Iterator for Axes<'a, D>
        where
            D: Dimension,
        {
            type Item = AxisDescription;
            fn next(&mut self) -> Option<Self::Item> {
                if self.start < self.end {
                    let i = self.start.post_inc();
                    Some(AxisDescription {
                        axis: Axis(i),
                        len: self.dim[i],
                        stride: self.strides[i] as Ixs,
                    })
                } else {
                    None
                }
            }
        }
        impl<'a, D> DoubleEndedIterator for Axes<'a, D>
        where
            D: Dimension,
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.start < self.end {
                    let i = self.end.pre_dec();
                    Some(AxisDescription {
                        axis: Axis(i),
                        len: self.dim[i],
                        stride: self.strides[i] as Ixs,
                    })
                } else {
                    None
                }
            }
        }
        trait IncOps: Copy {
            fn post_inc(&mut self) -> Self;
            fn post_dec(&mut self) -> Self;
            fn pre_dec(&mut self) -> Self;
        }
        impl IncOps for usize {
            #[inline(always)]
            fn post_inc(&mut self) -> Self {
                let x = *self;
                *self += 1;
                x
            }
            #[inline(always)]
            fn post_dec(&mut self) -> Self {
                let x = *self;
                *self -= 1;
                x
            }
            #[inline(always)]
            fn pre_dec(&mut self) -> Self {
                *self -= 1;
                *self
            }
        }
    }
    mod axis {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct Axis(pub usize);
        impl Axis {
            #[inline(always)]
            pub fn index(self) -> usize {
                self.0
            }
        }
    }
    pub(crate) mod broadcast {
        use crate::error::*;
        use crate::{Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
        pub(crate) fn co_broadcast<D1, D2, Output>(
            shape1: &D1,
            shape2: &D2,
        ) -> Result<Output, ShapeError>
        where
            D1: Dimension,
            D2: Dimension,
            Output: Dimension,
        {
            let (k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
            if overflow {
                return co_broadcast::<D2, D1, Output>(shape2, shape1);
            }
            let mut out = Output::zeros(shape1.ndim());
            for (out, s) in izip!(out.slice_mut(), shape1.slice()) {
                *out = *s;
            }
            for (out, s2) in izip!(&mut out.slice_mut()[k..], shape2.slice()) {
                if *out != *s2 {
                    if *out == 1 {
                        *out = *s2
                    } else if *s2 != 1 {
                        return Err(from_kind(ErrorKind::IncompatibleShape));
                    }
                }
            }
            Ok(out)
        }
        pub trait DimMax<Other: Dimension> {
            type Output: Dimension;
        }
        impl<D: Dimension> DimMax<D> for D {
            type Output = D;
        }
        macro_rules! impl_broadcast_distinct_fixed {
            ($ smaller : ty , $ larger : ty) => {
                impl DimMax<$larger> for $smaller {
                    type Output = $larger;
                }
                impl DimMax<$smaller> for $larger {
                    type Output = $larger;
                }
            };
        }
        impl_broadcast_distinct_fixed!(Ix0, Ix1);
        impl_broadcast_distinct_fixed!(Ix0, Ix2);
        impl_broadcast_distinct_fixed!(Ix0, Ix3);
        impl_broadcast_distinct_fixed!(Ix0, Ix4);
        impl_broadcast_distinct_fixed!(Ix0, Ix5);
        impl_broadcast_distinct_fixed!(Ix0, Ix6);
        impl_broadcast_distinct_fixed!(Ix1, Ix2);
        impl_broadcast_distinct_fixed!(Ix2, Ix3);
        impl_broadcast_distinct_fixed!(Ix3, Ix4);
        impl_broadcast_distinct_fixed!(Ix4, Ix5);
        impl_broadcast_distinct_fixed!(Ix5, Ix6);
        impl_broadcast_distinct_fixed!(Ix0, IxDyn);
        impl_broadcast_distinct_fixed!(Ix1, IxDyn);
        impl_broadcast_distinct_fixed!(Ix2, IxDyn);
        impl_broadcast_distinct_fixed!(Ix3, IxDyn);
        impl_broadcast_distinct_fixed!(Ix4, IxDyn);
        impl_broadcast_distinct_fixed!(Ix5, IxDyn);
        impl_broadcast_distinct_fixed!(Ix6, IxDyn);
    }
    mod conversion {
        use crate::{Dim, Dimension, Ix, Ix1, IxDyn, IxDynImpl};
        use alloc::vec::Vec;
        use num_traits::Zero;
        use std::ops::{Index, IndexMut};
        macro_rules ! index { ($ m : ident $ arg : tt 0) => ($ m ! ($ arg)) ; ($ m : ident $ arg : tt 1) => ($ m ! ($ arg 0)) ; ($ m : ident $ arg : tt 2) => ($ m ! ($ arg 0 1)) ; ($ m : ident $ arg : tt 3) => ($ m ! ($ arg 0 1 2)) ; ($ m : ident $ arg : tt 4) => ($ m ! ($ arg 0 1 2 3)) ; ($ m : ident $ arg : tt 5) => ($ m ! ($ arg 0 1 2 3 4)) ; ($ m : ident $ arg : tt 6) => ($ m ! ($ arg 0 1 2 3 4 5)) ; ($ m : ident $ arg : tt 7) => ($ m ! ($ arg 0 1 2 3 4 5 6)) ; }
        macro_rules ! index_item { ($ m : ident $ arg : tt 0) => () ; ($ m : ident $ arg : tt 1) => ($ m ! ($ arg 0) ;) ; ($ m : ident $ arg : tt 2) => ($ m ! ($ arg 0 1) ;) ; ($ m : ident $ arg : tt 3) => ($ m ! ($ arg 0 1 2) ;) ; ($ m : ident $ arg : tt 4) => ($ m ! ($ arg 0 1 2 3) ;) ; ($ m : ident $ arg : tt 5) => ($ m ! ($ arg 0 1 2 3 4) ;) ; ($ m : ident $ arg : tt 6) => ($ m ! ($ arg 0 1 2 3 4 5) ;) ; ($ m : ident $ arg : tt 7) => ($ m ! ($ arg 0 1 2 3 4 5 6) ;) ; }
        pub trait IntoDimension {
            type Dim: Dimension;
            fn into_dimension(self) -> Self::Dim;
        }
        impl IntoDimension for Ix {
            type Dim = Ix1;
            #[inline(always)]
            fn into_dimension(self) -> Ix1 {
                Ix1(self)
            }
        }
        impl<D> IntoDimension for D
        where
            D: Dimension,
        {
            type Dim = D;
            #[inline(always)]
            fn into_dimension(self) -> Self {
                self
            }
        }
        impl IntoDimension for IxDynImpl {
            type Dim = IxDyn;
            #[inline(always)]
            fn into_dimension(self) -> Self::Dim {
                Dim::new(self)
            }
        }
        impl IntoDimension for Vec<Ix> {
            type Dim = IxDyn;
            #[inline(always)]
            fn into_dimension(self) -> Self::Dim {
                Dim::new(IxDynImpl::from(self))
            }
        }
        pub trait Convert {
            type To;
            fn convert(self) -> Self::To;
        }
        macro_rules! sub {
            ($ _x : tt $ y : tt) => {
                $y
            };
        }
        macro_rules ! tuple_type { ([$ T : ident] $ ($ index : tt) *) => (($ (sub ! ($ index $ T) ,) *)) }
        macro_rules ! tuple_expr { ([$ self_ : expr] $ ($ index : tt) *) => (($ ($ self_ [$ index] ,) *)) }
        macro_rules ! array_expr { ([$ self_ : expr] $ ($ index : tt) *) => ([$ ($ self_ . $ index ,) *]) }
        macro_rules ! array_zero { ([] $ ($ index : tt) *) => ([$ (sub ! ($ index 0) ,) *]) }
        macro_rules ! tuple_to_array { ([] $ ($ n : tt) *) => { $ (impl Convert for [Ix ; $ n] { type To = index ! (tuple_type [Ix] $ n) ; # [inline] fn convert (self) -> Self :: To { index ! (tuple_expr [self] $ n) } } impl IntoDimension for [Ix ; $ n] { type Dim = Dim < [Ix ; $ n] >; # [inline (always)] fn into_dimension (self) -> Self :: Dim { Dim :: new (self) } } impl IntoDimension for index ! (tuple_type [Ix] $ n) { type Dim = Dim < [Ix ; $ n] >; # [inline (always)] fn into_dimension (self) -> Self :: Dim { Dim :: new (index ! (array_expr [self] $ n)) } } impl Index < usize > for Dim < [Ix ; $ n] > { type Output = usize ; # [inline (always)] fn index (& self , index : usize) -> & Self :: Output { & self . ix () [index] } } impl IndexMut < usize > for Dim < [Ix ; $ n] > { # [inline (always)] fn index_mut (& mut self , index : usize) -> & mut Self :: Output { & mut self . ixm () [index] } } impl Zero for Dim < [Ix ; $ n] > { # [inline] fn zero () -> Self { Dim :: new (index ! (array_zero [] $ n)) } fn is_zero (& self) -> bool { self . slice () . iter () . all (| x | * x == 0) } }) * } }
        index_item ! (tuple_to_array [] 7);
    }
    pub mod dim {
        use super::Dimension;
        use super::IntoDimension;
        use crate::itertools::zip;
        use crate::Ix;
        use std::fmt;
        #[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
        pub struct Dim<I: ?Sized> {
            index: I,
        }
        impl<I> Dim<I> {
            pub(crate) fn new(index: I) -> Dim<I> {
                Dim { index }
            }
            #[inline(always)]
            pub(crate) fn ix(&self) -> &I {
                &self.index
            }
            #[inline(always)]
            pub(crate) fn ixm(&mut self) -> &mut I {
                &mut self.index
            }
        }
        #[allow(non_snake_case)]
        pub fn Dim<T>(index: T) -> T::Dim
        where
            T: IntoDimension,
        {
            index.into_dimension()
        }
        impl<I> fmt::Debug for Dim<I>
        where
            I: fmt::Debug,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:?}", self.index)
            }
        }
        use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
        macro_rules! impl_op {
            ($ op : ident , $ op_m : ident , $ opassign : ident , $ opassign_m : ident , $ expr : ident) => {
                impl<I> $op for Dim<I>
                where
                    Dim<I>: Dimension,
                {
                    type Output = Self;
                    fn $op_m(mut self, rhs: Self) -> Self {
                        $expr!(self, &rhs);
                        self
                    }
                }
                impl<I> $opassign for Dim<I>
                where
                    Dim<I>: Dimension,
                {
                    fn $opassign_m(&mut self, rhs: Self) {
                        $expr!(*self, &rhs);
                    }
                }
                impl<'a, I> $opassign<&'a Dim<I>> for Dim<I>
                where
                    Dim<I>: Dimension,
                {
                    fn $opassign_m(&mut self, rhs: &Self) {
                        for (x, &y) in zip(self.slice_mut(), rhs.slice()) {
                            $expr!(*x, y);
                        }
                    }
                }
            };
        }
        macro_rules! impl_single_op {
            ($ op : ident , $ op_m : ident , $ opassign : ident , $ opassign_m : ident , $ expr : ident) => {
                impl $op<Ix> for Dim<[Ix; 1]> {
                    type Output = Self;
                    #[inline]
                    fn $op_m(mut self, rhs: Ix) -> Self {
                        $expr!(self, rhs);
                        self
                    }
                }
                impl $opassign<Ix> for Dim<[Ix; 1]> {
                    #[inline]
                    fn $opassign_m(&mut self, rhs: Ix) {
                        $expr!((*self)[0], rhs);
                    }
                }
            };
        }
        macro_rules! impl_scalar_op {
            ($ op : ident , $ op_m : ident , $ opassign : ident , $ opassign_m : ident , $ expr : ident) => {
                impl<I> $op<Ix> for Dim<I>
                where
                    Dim<I>: Dimension,
                {
                    type Output = Self;
                    fn $op_m(mut self, rhs: Ix) -> Self {
                        $expr!(self, rhs);
                        self
                    }
                }
                impl<I> $opassign<Ix> for Dim<I>
                where
                    Dim<I>: Dimension,
                {
                    fn $opassign_m(&mut self, rhs: Ix) {
                        for x in self.slice_mut() {
                            $expr!(*x, rhs);
                        }
                    }
                }
            };
        }
        macro_rules! add {
            ($ x : expr , $ y : expr) => {
                $x += $y;
            };
        }
        macro_rules! sub {
            ($ x : expr , $ y : expr) => {
                $x -= $y;
            };
        }
        macro_rules! mul {
            ($ x : expr , $ y : expr) => {
                $x *= $y;
            };
        }
        impl_op!(Add, add, AddAssign, add_assign, add);
        impl_op!(Sub, sub, SubAssign, sub_assign, sub);
        impl_op!(Mul, mul, MulAssign, mul_assign, mul);
        impl_scalar_op!(Mul, mul, MulAssign, mul_assign, mul);
    }
    mod dimension_trait {
        use super::axes_of;
        use super::conversion::Convert;
        use super::ops::DimAdd;
        use super::{stride_offset, stride_offset_checked};
        use crate::itertools::{enumerate, zip};
        use crate::IntoDimension;
        use crate::RemoveAxis;
        use crate::{ArrayView1, ArrayViewMut1};
        use crate::{Axis, DimMax};
        use crate::{Dim, Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, IxDynImpl, Ixs};
        use alloc::vec::Vec;
        use std::fmt::Debug;
        use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
        use std::ops::{Index, IndexMut};
        pub trait Dimension:
            Clone
            + Eq
            + Debug
            + Send
            + Sync
            + Default
            + IndexMut<usize, Output = usize>
            + Add<Self, Output = Self>
            + AddAssign
            + for<'x> AddAssign<&'x Self>
            + Sub<Self, Output = Self>
            + SubAssign
            + for<'x> SubAssign<&'x Self>
            + Mul<usize, Output = Self>
            + Mul<Self, Output = Self>
            + MulAssign
            + for<'x> MulAssign<&'x Self>
            + MulAssign<usize>
            + DimMax<Ix0, Output = Self>
            + DimMax<Self, Output = Self>
            + DimMax<IxDyn, Output = IxDyn>
            + DimMax<<Self as Dimension>::Smaller, Output = Self>
            + DimMax<<Self as Dimension>::Larger, Output = <Self as Dimension>::Larger>
            + DimAdd<Self>
            + DimAdd<<Self as Dimension>::Smaller>
            + DimAdd<<Self as Dimension>::Larger>
            + DimAdd<Ix0, Output = Self>
            + DimAdd<Ix1, Output = <Self as Dimension>::Larger>
            + DimAdd<IxDyn, Output = IxDyn>
        {
            const NDIM: Option<usize>;
            type Pattern: IntoDimension<Dim = Self> + Clone + Debug + PartialEq + Eq + Default;
            type Smaller: Dimension;
            type Larger: Dimension + RemoveAxis;
            fn ndim(&self) -> usize;
            fn into_pattern(self) -> Self::Pattern;
            fn size(&self) -> usize {
                self.slice().iter().fold(1, |s, &a| s * a as usize)
            }
            fn size_checked(&self) -> Option<usize> {
                self.slice()
                    .iter()
                    .fold(Some(1), |s, &a| s.and_then(|s_| s_.checked_mul(a)))
            }
            #[doc(hidden)]
            fn slice(&self) -> &[Ix];
            #[doc(hidden)]
            fn slice_mut(&mut self) -> &mut [Ix];
            #[doc = " Borrow as a read-only array view."]
            fn as_array_view(&self) -> ArrayView1<'_, Ix> {
                ArrayView1::from(self.slice())
            }
            #[doc = " Borrow as a read-write array view."]
            fn as_array_view_mut(&mut self) -> ArrayViewMut1<'_, Ix> {
                ArrayViewMut1::from(self.slice_mut())
            }
            #[doc(hidden)]
            fn equal(&self, rhs: &Self) -> bool {
                self.slice() == rhs.slice()
            }
            #[doc = " Returns the strides for a standard layout array with the given shape."]
            #[doc = ""]
            #[doc = " If the array is non-empty, the strides result in contiguous layout; if"]
            #[doc = " the array is empty, the strides are all zeros."]
            #[doc(hidden)]
            fn default_strides(&self) -> Self {
                let mut strides = Self::zeros(self.ndim());
                if self.slice().iter().all(|&d| d != 0) {
                    let mut it = strides.slice_mut().iter_mut().rev();
                    if let Some(rs) = it.next() {
                        *rs = 1;
                    }
                    let mut cum_prod = 1;
                    for (rs, dim) in it.zip(self.slice().iter().rev()) {
                        cum_prod *= *dim;
                        *rs = cum_prod;
                    }
                }
                strides
            }
            #[doc = " Returns the strides for a Fortran layout array with the given shape."]
            #[doc = ""]
            #[doc = " If the array is non-empty, the strides result in contiguous layout; if"]
            #[doc = " the array is empty, the strides are all zeros."]
            #[doc(hidden)]
            fn fortran_strides(&self) -> Self {
                let mut strides = Self::zeros(self.ndim());
                if self.slice().iter().all(|&d| d != 0) {
                    let mut it = strides.slice_mut().iter_mut();
                    if let Some(rs) = it.next() {
                        *rs = 1;
                    }
                    let mut cum_prod = 1;
                    for (rs, dim) in it.zip(self.slice()) {
                        cum_prod *= *dim;
                        *rs = cum_prod;
                    }
                }
                strides
            }
            #[doc = " Creates a dimension of all zeros with the specified ndim."]
            #[doc = ""]
            #[doc = " This method is useful for generalizing over fixed-size and"]
            #[doc = " variable-size dimension representations."]
            #[doc = ""]
            #[doc = " **Panics** if `Self` has a fixed size that is not `ndim`."]
            fn zeros(ndim: usize) -> Self;
            #[doc(hidden)]
            #[inline]
            fn first_index(&self) -> Option<Self> {
                for ax in self.slice().iter() {
                    if *ax == 0 {
                        return None;
                    }
                }
                Some(Self::zeros(self.ndim()))
            }
            #[doc(hidden)]
            #[doc = " Iteration -- Use self as size, and return next index after `index`"]
            #[doc = " or None if there are no more."]
            #[inline]
            fn next_for(&self, index: Self) -> Option<Self> {
                let mut index = index;
                let mut done = false;
                for (&dim, ix) in zip(self.slice(), index.slice_mut()).rev() {
                    *ix += 1;
                    if *ix == dim {
                        *ix = 0;
                    } else {
                        done = true;
                        break;
                    }
                }
                if done {
                    Some(index)
                } else {
                    None
                }
            }
            #[doc(hidden)]
            #[doc = " Iteration -- Use self as size, and create the next index after `index`"]
            #[doc = " Return false if iteration is done"]
            #[doc = ""]
            #[doc = " Next in f-order"]
            #[inline]
            fn next_for_f(&self, index: &mut Self) -> bool {
                let mut end_iteration = true;
                for (&dim, ix) in zip(self.slice(), index.slice_mut()) {
                    *ix += 1;
                    if *ix == dim {
                        *ix = 0;
                    } else {
                        end_iteration = false;
                        break;
                    }
                }
                !end_iteration
            }
            #[doc = " Returns `true` iff `strides1` and `strides2` are equivalent for the"]
            #[doc = " shape `self`."]
            #[doc = ""]
            #[doc = " The strides are equivalent if, for each axis with length > 1, the"]
            #[doc = " strides are equal."]
            #[doc = ""]
            #[doc = " Note: Returns `false` if any of the ndims don't match."]
            #[doc(hidden)]
            fn strides_equivalent<D>(&self, strides1: &Self, strides2: &D) -> bool
            where
                D: Dimension,
            {
                let shape_ndim = self.ndim();
                shape_ndim == strides1.ndim()
                    && shape_ndim == strides2.ndim()
                    && izip!(self.slice(), strides1.slice(), strides2.slice())
                        .all(|(&d, &s1, &s2)| d <= 1 || s1 as isize == s2 as isize)
            }
            #[doc(hidden)]
            #[doc = " Return stride offset for index."]
            fn stride_offset(index: &Self, strides: &Self) -> isize {
                let mut offset = 0;
                for (&i, &s) in izip!(index.slice(), strides.slice()) {
                    offset += stride_offset(i, s);
                }
                offset
            }
            #[doc(hidden)]
            #[doc = " Return stride offset for this dimension and index."]
            fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize> {
                stride_offset_checked(self.slice(), strides.slice(), index.slice())
            }
            #[doc(hidden)]
            fn last_elem(&self) -> usize {
                if self.ndim() == 0 {
                    0
                } else {
                    self.slice()[self.ndim() - 1]
                }
            }
            #[doc(hidden)]
            fn set_last_elem(&mut self, i: usize) {
                let nd = self.ndim();
                self.slice_mut()[nd - 1] = i;
            }
            #[doc(hidden)]
            fn is_contiguous(dim: &Self, strides: &Self) -> bool {
                let defaults = dim.default_strides();
                if strides.equal(&defaults) {
                    return true;
                }
                if dim.ndim() == 1 {
                    return strides[0] as isize == -1;
                }
                let order = strides._fastest_varying_stride_order();
                let strides = strides.slice();
                let dim_slice = dim.slice();
                let mut cstride = 1;
                for &i in order.slice() {
                    if dim_slice[i] != 1 && (strides[i] as isize).unsigned_abs() != cstride {
                        return false;
                    }
                    cstride *= dim_slice[i];
                }
                true
            }
            #[doc = " Return the axis ordering corresponding to the fastest variation"]
            #[doc = " (in ascending order)."]
            #[doc = ""]
            #[doc = " Assumes that no stride value appears twice."]
            #[doc(hidden)]
            fn _fastest_varying_stride_order(&self) -> Self {
                let mut indices = self.clone();
                for (i, elt) in enumerate(indices.slice_mut()) {
                    *elt = i;
                }
                let strides = self.slice();
                indices
                    .slice_mut()
                    .sort_by_key(|&i| (strides[i] as isize).abs());
                indices
            }
            #[doc = " Compute the minimum stride axis (absolute value), under the constraint"]
            #[doc = " that the length of the axis is > 1;"]
            #[doc(hidden)]
            fn min_stride_axis(&self, strides: &Self) -> Axis {
                let n = match self.ndim() {
                    0 => panic!("min_stride_axis: Array must have ndim > 0"),
                    1 => return Axis(0),
                    n => n,
                };
                axes_of(self, strides)
                    .rev()
                    .min_by_key(|ax| ax.stride.abs())
                    .map_or(Axis(n - 1), |ax| ax.axis)
            }
            #[doc = " Compute the maximum stride axis (absolute value), under the constraint"]
            #[doc = " that the length of the axis is > 1;"]
            #[doc(hidden)]
            fn max_stride_axis(&self, strides: &Self) -> Axis {
                match self.ndim() {
                    0 => panic!("max_stride_axis: Array must have ndim > 0"),
                    1 => return Axis(0),
                    _ => {}
                }
                axes_of(self, strides)
                    .filter(|ax| ax.len > 1)
                    .max_by_key(|ax| ax.stride.abs())
                    .map_or(Axis(0), |ax| ax.axis)
            }
            #[doc = " Convert the dimensional into a dynamic dimensional (IxDyn)."]
            fn into_dyn(self) -> IxDyn {
                IxDyn(self.slice())
            }
            #[doc(hidden)]
            fn from_dimension<D2: Dimension>(d: &D2) -> Option<Self> {
                let mut s = Self::default();
                if s.ndim() == d.ndim() {
                    for i in 0..d.ndim() {
                        s[i] = d[i];
                    }
                    Some(s)
                } else {
                    None
                }
            }
            #[doc(hidden)]
            fn insert_axis(&self, axis: Axis) -> Self::Larger;
            #[doc(hidden)]
            fn try_remove_axis(&self, axis: Axis) -> Self::Smaller;
            private_decl! {}
        }
        macro_rules ! impl_insert_axis_array (($ n : expr) => (# [inline] fn insert_axis (& self , axis : Axis) -> Self :: Larger { debug_assert ! (axis . index () <= $ n) ; let mut out = [1 ; $ n + 1] ; out [0 .. axis . index ()] . copy_from_slice (& self . slice () [0 .. axis . index ()]) ; out [axis . index () + 1 ..=$ n] . copy_from_slice (& self . slice () [axis . index () ..$ n]) ; Dim (out) }) ;) ;
        impl Dimension for Dim<[Ix; 0]> {
            const NDIM: Option<usize> = Some(0);
            type Pattern = ();
            type Smaller = Self;
            type Larger = Ix1;
            #[inline]
            fn ndim(&self) -> usize {
                0
            }
            #[inline]
            fn slice(&self) -> &[Ix] {
                &[]
            }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] {
                &mut []
            }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {}
            #[inline]
            fn zeros(ndim: usize) -> Self {
                assert_eq!(ndim, 0);
                Self::default()
            }
            impl_insert_axis_array!(0);
            #[inline]
            fn try_remove_axis(&self, _ignore: Axis) -> Self::Smaller {
                *self
            }
            private_impl! {}
        }
        impl Dimension for Dim<[Ix; 1]> {
            const NDIM: Option<usize> = Some(1);
            type Pattern = Ix;
            type Smaller = Ix0;
            type Larger = Ix2;
            #[inline]
            fn ndim(&self) -> usize {
                1
            }
            #[inline]
            fn slice(&self) -> &[Ix] {
                self.ix()
            }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] {
                self.ixm()
            }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {
                get!(&self, 0)
            }
            #[inline]
            fn zeros(ndim: usize) -> Self {
                assert_eq!(ndim, 1);
                Self::default()
            }
            impl_insert_axis_array!(1);
            #[inline]
            fn try_remove_axis(&self, axis: Axis) -> Self::Smaller {
                self.remove_axis(axis)
            }
            private_impl! {}
        }
        impl Dimension for Dim<[Ix; 2]> {
            const NDIM: Option<usize> = Some(2);
            type Pattern = (Ix, Ix);
            type Smaller = Ix1;
            type Larger = Ix3;
            #[inline]
            fn ndim(&self) -> usize {
                2
            }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {
                self.ix().convert()
            }
            #[inline]
            fn slice(&self) -> &[Ix] {
                self.ix()
            }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] {
                self.ixm()
            }
            #[inline]
            fn zeros(ndim: usize) -> Self {
                assert_eq!(ndim, 2);
                Self::default()
            }
            impl_insert_axis_array!(2);
            #[inline]
            fn try_remove_axis(&self, axis: Axis) -> Self::Smaller {
                self.remove_axis(axis)
            }
            private_impl! {}
        }
        impl Dimension for Dim<[Ix; 3]> {
            const NDIM: Option<usize> = Some(3);
            type Pattern = (Ix, Ix, Ix);
            type Smaller = Ix2;
            type Larger = Ix4;
            #[inline]
            fn ndim(&self) -> usize {
                3
            }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {
                self.ix().convert()
            }
            #[inline]
            fn slice(&self) -> &[Ix] {
                self.ix()
            }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] {
                self.ixm()
            }
            #[inline]
            fn zeros(ndim: usize) -> Self {
                assert_eq!(ndim, 3);
                Self::default()
            }
            impl_insert_axis_array!(3);
            #[inline]
            fn try_remove_axis(&self, axis: Axis) -> Self::Smaller {
                self.remove_axis(axis)
            }
            private_impl! {}
        }
        macro_rules ! large_dim { ($ n : expr , $ name : ident , $ pattern : ty , $ larger : ty , { $ ($ insert_axis : tt) * }) => (impl Dimension for Dim < [Ix ; $ n] > { const NDIM : Option < usize > = Some ($ n) ; type Pattern = $ pattern ; type Smaller = Dim < [Ix ; $ n - 1] >; type Larger = $ larger ; # [inline] fn ndim (& self) -> usize { $ n } # [inline] fn into_pattern (self) -> Self :: Pattern { self . ix () . convert () } # [inline] fn slice (& self) -> & [Ix] { self . ix () } # [inline] fn slice_mut (& mut self) -> & mut [Ix] { self . ixm () } # [inline] fn zeros (ndim : usize) -> Self { assert_eq ! (ndim , $ n) ; Self :: default () } $ ($ insert_axis) * # [inline] fn try_remove_axis (& self , axis : Axis) -> Self :: Smaller { self . remove_axis (axis) } private_impl ! { } }) }
        large_dim!(4, Ix4, (Ix, Ix, Ix, Ix), Ix5, {
            impl_insert_axis_array!(4);
        });
        large_dim!(5, Ix5, (Ix, Ix, Ix, Ix, Ix), Ix6, {
            impl_insert_axis_array!(5);
        });
        large_dim!(6, Ix6, (Ix, Ix, Ix, Ix, Ix, Ix), IxDyn, {
            fn insert_axis(&self, axis: Axis) -> Self::Larger {
                debug_assert!(axis.index() <= self.ndim());
                let mut out = Vec::with_capacity(self.ndim() + 1);
                out.extend_from_slice(&self.slice()[0..axis.index()]);
                out.push(1);
                out.extend_from_slice(&self.slice()[axis.index()..self.ndim()]);
                Dim(out)
            }
        });
        #[doc = " IxDyn is a \"dynamic\" index, pretty hard to use when indexing,"]
        #[doc = " and memory wasteful, but it allows an arbitrary and dynamic number of axes."]
        impl Dimension for IxDyn {
            const NDIM: Option<usize> = None;
            type Pattern = Self;
            type Smaller = Self;
            type Larger = Self;
            #[inline]
            fn ndim(&self) -> usize {
                self.ix().len()
            }
            #[inline]
            fn slice(&self) -> &[Ix] {
                self.ix()
            }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] {
                self.ixm()
            }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {
                self
            }
            #[inline]
            fn zeros(ndim: usize) -> Self {
                IxDyn::zeros(ndim)
            }
            #[inline]
            fn insert_axis(&self, axis: Axis) -> Self::Larger {
                debug_assert!(axis.index() <= self.ndim());
                Dim::new(self.ix().insert(axis.index()))
            }
            #[inline]
            fn try_remove_axis(&self, axis: Axis) -> Self::Smaller {
                if self.ndim() > 0 {
                    self.remove_axis(axis)
                } else {
                    self.clone()
                }
            }
            private_impl! {}
        }
        impl Index<usize> for Dim<IxDynImpl> {
            type Output = <IxDynImpl as Index<usize>>::Output;
            fn index(&self, index: usize) -> &Self::Output {
                &self.ix()[index]
            }
        }
        impl IndexMut<usize> for Dim<IxDynImpl> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.ixm()[index]
            }
        }
    }
    mod dynindeximpl {
        use crate::imp_prelude::*;
        use alloc::boxed::Box;
        use alloc::vec;
        use alloc::vec::Vec;
        use std::hash::{Hash, Hasher};
        use std::ops::{Deref, DerefMut, Index, IndexMut};
        const CAP: usize = 4;
        #[doc = " T is usize or isize"]
        #[derive(Debug)]
        enum IxDynRepr<T> {
            Inline(u32, [T; CAP]),
            Alloc(Box<[T]>),
        }
        impl<T> Deref for IxDynRepr<T> {
            type Target = [T];
            fn deref(&self) -> &[T] {
                match *self {
                    IxDynRepr::Inline(len, ref ar) => {
                        debug_assert!(len as usize <= ar.len());
                        unsafe { ar.get_unchecked(..len as usize) }
                    }
                    IxDynRepr::Alloc(ref ar) => ar,
                }
            }
        }
        impl<T> DerefMut for IxDynRepr<T> {
            fn deref_mut(&mut self) -> &mut [T] {
                match *self {
                    IxDynRepr::Inline(len, ref mut ar) => {
                        debug_assert!(len as usize <= ar.len());
                        unsafe { ar.get_unchecked_mut(..len as usize) }
                    }
                    IxDynRepr::Alloc(ref mut ar) => ar,
                }
            }
        }
        #[doc = " The default is equivalent to `Self::from(&[0])`."]
        impl Default for IxDynRepr<Ix> {
            fn default() -> Self {
                Self::copy_from(&[0])
            }
        }
        use num_traits::Zero;
        impl<T: Copy + Zero> IxDynRepr<T> {
            pub fn copy_from(x: &[T]) -> Self {
                if x.len() <= CAP {
                    let mut arr = [T::zero(); CAP];
                    arr[..x.len()].copy_from_slice(x);
                    IxDynRepr::Inline(x.len() as _, arr)
                } else {
                    Self::from(x)
                }
            }
        }
        impl<T: Copy + Zero> IxDynRepr<T> {
            fn from_vec_auto(v: Vec<T>) -> Self {
                if v.len() <= CAP {
                    Self::copy_from(&v)
                } else {
                    Self::from_vec(v)
                }
            }
        }
        impl<T: Copy> IxDynRepr<T> {
            fn from_vec(v: Vec<T>) -> Self {
                IxDynRepr::Alloc(v.into_boxed_slice())
            }
            fn from(x: &[T]) -> Self {
                Self::from_vec(x.to_vec())
            }
        }
        impl<T: Copy> Clone for IxDynRepr<T> {
            fn clone(&self) -> Self {
                match *self {
                    IxDynRepr::Inline(len, arr) => IxDynRepr::Inline(len, arr),
                    _ => Self::from(&self[..]),
                }
            }
        }
        impl<T: Eq> Eq for IxDynRepr<T> {}
        impl<T: PartialEq> PartialEq for IxDynRepr<T> {
            fn eq(&self, rhs: &Self) -> bool {
                match (self, rhs) {
                    (&IxDynRepr::Inline(slen, ref sarr), &IxDynRepr::Inline(rlen, ref rarr)) => {
                        slen == rlen
                            && (0..CAP as usize)
                                .filter(|&i| i < slen as usize)
                                .all(|i| sarr[i] == rarr[i])
                    }
                    _ => self[..] == rhs[..],
                }
            }
        }
        impl<T: Hash> Hash for IxDynRepr<T> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                Hash::hash(&self[..], state)
            }
        }
        #[doc = " Dynamic dimension or index type."]
        #[doc = ""]
        #[doc = " Use `IxDyn` directly. This type implements a dynamic number of"]
        #[doc = " dimensions or indices. Short dimensions are stored inline and don't need"]
        #[doc = " any dynamic memory allocation."]
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
        pub struct IxDynImpl(IxDynRepr<Ix>);
        impl IxDynImpl {
            pub(crate) fn insert(&self, i: usize) -> Self {
                let len = self.len();
                debug_assert!(i <= len);
                IxDynImpl(if len < CAP {
                    let mut out = [1; CAP];
                    out[0..i].copy_from_slice(&self[0..i]);
                    out[i + 1..=len].copy_from_slice(&self[i..len]);
                    IxDynRepr::Inline((len + 1) as u32, out)
                } else {
                    let mut out = Vec::with_capacity(len + 1);
                    out.extend_from_slice(&self[0..i]);
                    out.push(1);
                    out.extend_from_slice(&self[i..len]);
                    IxDynRepr::from_vec(out)
                })
            }
            fn remove(&self, i: usize) -> Self {
                IxDynImpl(match self.0 {
                    IxDynRepr::Inline(0, _) => IxDynRepr::Inline(0, [0; CAP]),
                    IxDynRepr::Inline(1, _) => IxDynRepr::Inline(0, [0; CAP]),
                    IxDynRepr::Inline(2, ref arr) => {
                        let mut out = [0; CAP];
                        out[0] = arr[1 - i];
                        IxDynRepr::Inline(1, out)
                    }
                    ref ixdyn => {
                        let len = ixdyn.len();
                        let mut result = IxDynRepr::copy_from(&ixdyn[..len - 1]);
                        for j in i..len - 1 {
                            result[j] = ixdyn[j + 1]
                        }
                        result
                    }
                })
            }
        }
        impl<'a> From<&'a [Ix]> for IxDynImpl {
            #[inline]
            fn from(ix: &'a [Ix]) -> Self {
                IxDynImpl(IxDynRepr::copy_from(ix))
            }
        }
        impl From<Vec<Ix>> for IxDynImpl {
            #[inline]
            fn from(ix: Vec<Ix>) -> Self {
                IxDynImpl(IxDynRepr::from_vec_auto(ix))
            }
        }
        impl<J> Index<J> for IxDynImpl
        where
            [Ix]: Index<J>,
        {
            type Output = <[Ix] as Index<J>>::Output;
            fn index(&self, index: J) -> &Self::Output {
                &self.0[index]
            }
        }
        impl<J> IndexMut<J> for IxDynImpl
        where
            [Ix]: IndexMut<J>,
        {
            fn index_mut(&mut self, index: J) -> &mut Self::Output {
                &mut self.0[index]
            }
        }
        impl Deref for IxDynImpl {
            type Target = [Ix];
            #[inline]
            fn deref(&self) -> &[Ix] {
                &self.0
            }
        }
        impl DerefMut for IxDynImpl {
            #[inline]
            fn deref_mut(&mut self) -> &mut [Ix] {
                &mut self.0
            }
        }
        impl<'a> IntoIterator for &'a IxDynImpl {
            type Item = &'a Ix;
            type IntoIter = <&'a [Ix] as IntoIterator>::IntoIter;
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self[..].iter()
            }
        }
        impl RemoveAxis for Dim<IxDynImpl> {
            fn remove_axis(&self, axis: Axis) -> Self {
                debug_assert!(axis.index() < self.ndim());
                Dim::new(self.ix().remove(axis.index()))
            }
        }
        impl IxDyn {
            #[doc = " Create a new dimension value with `n` axes, all zeros"]
            #[inline]
            pub fn zeros(n: usize) -> IxDyn {
                const ZEROS: &[usize] = &[0; 4];
                if n <= ZEROS.len() {
                    Dim(&ZEROS[..n])
                } else {
                    Dim(vec![0; n])
                }
            }
        }
    }
    mod ndindex {
        use super::{stride_offset, stride_offset_checked};
        use crate::itertools::zip;
        use crate::{
            Dim, Dimension, IntoDimension, Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, IxDynImpl,
        };
        use std::fmt::Debug;
        #[doc = " Tuple or fixed size arrays that can be used to index an array."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::arr2;"]
        #[doc = ""]
        #[doc = " let mut a = arr2(&[[0, 1],"]
        #[doc = "                    [2, 3]]);"]
        #[doc = " assert_eq!(a[[0, 1]], 1);"]
        #[doc = " assert_eq!(a[[1, 1]], 3);"]
        #[doc = " a[[1, 1]] += 1;"]
        #[doc = " assert_eq!(a[(1, 1)], 4);"]
        #[doc = " ```"]
        #[allow(clippy::missing_safety_doc)]
        pub unsafe trait NdIndex<E>: Debug {
            #[doc(hidden)]
            fn index_checked(&self, dim: &E, strides: &E) -> Option<isize>;
            #[doc(hidden)]
            fn index_unchecked(&self, strides: &E) -> isize;
        }
        unsafe impl<D> NdIndex<D> for D
        where
            D: Dimension,
        {
            fn index_checked(&self, dim: &D, strides: &D) -> Option<isize> {
                dim.stride_offset_checked(strides, self)
            }
            fn index_unchecked(&self, strides: &D) -> isize {
                D::stride_offset(self, strides)
            }
        }
        unsafe impl NdIndex<Ix1> for Ix {
            #[inline]
            fn index_checked(&self, dim: &Ix1, strides: &Ix1) -> Option<isize> {
                dim.stride_offset_checked(strides, &Ix1(*self))
            }
            #[inline(always)]
            fn index_unchecked(&self, strides: &Ix1) -> isize {
                stride_offset(*self, get!(strides, 0))
            }
        }
        macro_rules ! ndindex_with_array { ($ ([$ n : expr , $ ix_n : ident $ ($ index : tt) *]) +) => { $ (unsafe impl NdIndex <$ ix_n > for [Ix ; $ n] { # [inline] fn index_checked (& self , dim : &$ ix_n , strides : &$ ix_n) -> Option < isize > { dim . stride_offset_checked (strides , & self . into_dimension ()) } # [inline] fn index_unchecked (& self , _strides : &$ ix_n) -> isize { $ (stride_offset (self [$ index] , get ! (_strides , $ index)) +) * 0 } }) + } ; }
        impl<'a> IntoDimension for &'a [Ix] {
            type Dim = IxDyn;
            fn into_dimension(self) -> Self::Dim {
                Dim(IxDynImpl::from(self))
            }
        }
    }
    mod ops {
        use crate::imp_prelude::*;
        #[doc = " Adds the two dimensions at compile time."]
        pub trait DimAdd<D: Dimension> {
            #[doc = " The sum of the two dimensions."]
            type Output: Dimension;
        }
        macro_rules! impl_dimadd_const_out_const {
            ($ lhs : expr , $ rhs : expr) => {
                impl DimAdd<Dim<[usize; $rhs]>> for Dim<[usize; $lhs]> {
                    type Output = Dim<[usize; $lhs + $rhs]>;
                }
            };
        }
        macro_rules! impl_dimadd_const_out_dyn {
            ($ lhs : expr , IxDyn) => {
                impl DimAdd<IxDyn> for Dim<[usize; $lhs]> {
                    type Output = IxDyn;
                }
            };
            ($ lhs : expr , $ rhs : expr) => {
                impl DimAdd<Dim<[usize; $rhs]>> for Dim<[usize; $lhs]> {
                    type Output = IxDyn;
                }
            };
        }
        impl<D: Dimension> DimAdd<D> for Ix0 {
            type Output = D;
        }
        impl_dimadd_const_out_const!(1, 0);
        impl_dimadd_const_out_const!(1, 1);
        impl_dimadd_const_out_const!(1, 2);
        impl_dimadd_const_out_dyn!(1, IxDyn);
        impl_dimadd_const_out_const!(2, 0);
        impl_dimadd_const_out_const!(2, 1);
        impl_dimadd_const_out_const!(2, 2);
        impl_dimadd_const_out_const!(2, 3);
        impl_dimadd_const_out_dyn!(2, IxDyn);
        impl_dimadd_const_out_const!(3, 0);
        impl_dimadd_const_out_const!(3, 1);
        impl_dimadd_const_out_const!(3, 2);
        impl_dimadd_const_out_const!(3, 3);
        impl_dimadd_const_out_dyn!(3, 4);
        impl_dimadd_const_out_dyn!(3, IxDyn);
        impl_dimadd_const_out_const!(4, 0);
        impl_dimadd_const_out_const!(4, 1);
        impl_dimadd_const_out_dyn!(4, 3);
        impl_dimadd_const_out_dyn!(4, 4);
        impl_dimadd_const_out_dyn!(4, 5);
        impl_dimadd_const_out_dyn!(4, IxDyn);
        impl_dimadd_const_out_const!(5, 0);
        impl_dimadd_const_out_const!(5, 1);
        impl_dimadd_const_out_dyn!(5, 4);
        impl_dimadd_const_out_dyn!(5, 5);
        impl_dimadd_const_out_dyn!(5, 6);
        impl_dimadd_const_out_dyn!(5, IxDyn);
        impl_dimadd_const_out_const!(6, 0);
        impl_dimadd_const_out_dyn!(6, 1);
        impl_dimadd_const_out_dyn!(6, 5);
        impl_dimadd_const_out_dyn!(6, 6);
        impl_dimadd_const_out_dyn!(6, IxDyn);
        impl<D: Dimension> DimAdd<D> for IxDyn {
            type Output = IxDyn;
        }
    }
    mod remove_axis {
        use crate::{Axis, Dim, Dimension, Ix, Ix0, Ix1};
        #[doc = " Array shape with a next smaller dimension."]
        #[doc = ""]
        #[doc = " `RemoveAxis` defines a larger-than relation for array shapes:"]
        #[doc = " removing one axis from *Self* gives smaller dimension *Smaller*."]
        pub trait RemoveAxis: Dimension {
            fn remove_axis(&self, axis: Axis) -> Self::Smaller;
        }
        impl RemoveAxis for Dim<[Ix; 1]> {
            #[inline]
            fn remove_axis(&self, axis: Axis) -> Ix0 {
                debug_assert!(axis.index() < self.ndim());
                Ix0()
            }
        }
        impl RemoveAxis for Dim<[Ix; 2]> {
            #[inline]
            fn remove_axis(&self, axis: Axis) -> Ix1 {
                let axis = axis.index();
                debug_assert!(axis < self.ndim());
                if axis == 0 {
                    Ix1(get!(self, 1))
                } else {
                    Ix1(get!(self, 0))
                }
            }
        }
        macro_rules ! impl_remove_axis_array (($ ($ n : expr) ,*) => ($ (impl RemoveAxis for Dim < [Ix ; $ n] > { # [inline] fn remove_axis (& self , axis : Axis) -> Self :: Smaller { debug_assert ! (axis . index () < self . ndim ()) ; let mut result = Dim ([0 ; $ n - 1]) ; { let src = self . slice () ; let dst = result . slice_mut () ; dst [.. axis . index ()] . copy_from_slice (& src [.. axis . index ()]) ; dst [axis . index () ..] . copy_from_slice (& src [axis . index () + 1 ..]) ; } result } }) *) ;) ;
        impl_remove_axis_array!(3, 4, 5, 6);
    }
    pub(crate) mod reshape {
        use crate::dimension::sequence::{Forward, Reverse, Sequence, SequenceMut};
        use crate::{Dimension, ErrorKind, Order, ShapeError};
        #[inline]
        pub(crate) fn reshape_dim<D, E>(
            from: &D,
            strides: &D,
            to: &E,
            order: Order,
        ) -> Result<E, ShapeError>
        where
            D: Dimension,
            E: Dimension,
        {
            debug_assert_eq!(from.ndim(), strides.ndim());
            let mut to_strides = E::zeros(to.ndim());
            match order {
                Order::RowMajor => {
                    reshape_dim_c(
                        &Forward(from),
                        &Forward(strides),
                        &Forward(to),
                        Forward(&mut to_strides),
                    )?;
                }
                Order::ColumnMajor => {
                    reshape_dim_c(
                        &Reverse(from),
                        &Reverse(strides),
                        &Reverse(to),
                        Reverse(&mut to_strides),
                    )?;
                }
            }
            Ok(to_strides)
        }
        #[doc = " Try to reshape an array with dimensions `from_dim` and strides `from_strides` to the new"]
        #[doc = " dimension `to_dim`, while keeping the same layout of elements in memory. The strides needed"]
        #[doc = " if this is possible are stored into `to_strides`."]
        #[doc = ""]
        #[doc = " This function uses RowMajor index ordering if the inputs are read in the forward direction"]
        #[doc = " (index 0 is axis 0 etc) and ColumnMajor index ordering if the inputs are read in reversed"]
        #[doc = " direction (as made possible with the Sequence trait)."]
        #[doc = " "]
        #[doc = " Preconditions:"]
        #[doc = ""]
        #[doc = " 1. from_dim and to_dim are valid dimensions (product of all non-zero axes"]
        #[doc = " fits in isize::MAX)."]
        #[doc = " 2. from_dim and to_dim are don't have any axes that are zero (that should be handled before"]
        #[doc = "    this function)."]
        #[doc = " 3. `to_strides` should be an all-zeros or all-ones dimension of the right dimensionality"]
        #[doc = " (but it will be overwritten after successful exit of this function)."]
        #[doc = ""]
        #[doc = " This function returns:"]
        #[doc = ""]
        #[doc = " - IncompatibleShape if the two shapes are not of matching number of elements"]
        #[doc = " - IncompatibleLayout if the input shape and stride can not be remapped to the output shape"]
        #[doc = "   without moving the array data into a new memory layout."]
        #[doc = " - Ok if the from dim could be mapped to the new to dim."]
        fn reshape_dim_c<D, E, E2>(
            from_dim: &D,
            from_strides: &D,
            to_dim: &E,
            mut to_strides: E2,
        ) -> Result<(), ShapeError>
        where
            D: Sequence<Output = usize>,
            E: Sequence<Output = usize>,
            E2: SequenceMut<Output = usize>,
        {
            let mut fi = 0;
            let mut ti = 0;
            while fi < from_dim.len() && ti < to_dim.len() {
                let mut fd = from_dim[fi];
                let mut fs = from_strides[fi] as isize;
                let mut td = to_dim[ti];
                if fd == td {
                    to_strides[ti] = from_strides[fi];
                    fi += 1;
                    ti += 1;
                    continue;
                }
                if fd == 1 {
                    fi += 1;
                    continue;
                }
                if td == 1 {
                    to_strides[ti] = 1;
                    ti += 1;
                    continue;
                }
                if fd == 0 || td == 0 {
                    debug_assert!(false, "zero dim not handled by this function");
                    return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
                }
                let mut fstride_whole = fs * (fd as isize);
                let mut fd_product = fd;
                let mut td_product = td;
                while fd_product != td_product {
                    if fd_product < td_product {
                        fi += 1;
                        if fi >= from_dim.len() {
                            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
                        }
                        fd = from_dim[fi];
                        fd_product *= fd;
                        if fd > 1 {
                            let fs_old = fs;
                            fs = from_strides[fi] as isize;
                            if fs_old != fd as isize * fs {
                                return Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout));
                            }
                        }
                    } else {
                        fstride_whole /= td as isize;
                        to_strides[ti] = fstride_whole as usize;
                        ti += 1;
                        if ti >= to_dim.len() {
                            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
                        }
                        td = to_dim[ti];
                        td_product *= td;
                    }
                }
                fstride_whole /= td as isize;
                to_strides[ti] = fstride_whole as usize;
                fi += 1;
                ti += 1;
            }
            while fi < from_dim.len() && from_dim[fi] == 1 {
                fi += 1;
            }
            while ti < to_dim.len() && to_dim[ti] == 1 {
                to_strides[ti] = 1;
                ti += 1;
            }
            if fi < from_dim.len() || ti < to_dim.len() {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
            Ok(())
        }
    }
    mod sequence {
        use crate::dimension::Dimension;
        use std::ops::Index;
        use std::ops::IndexMut;
        pub(in crate::dimension) struct Forward<D>(pub(crate) D);
        pub(in crate::dimension) struct Reverse<D>(pub(crate) D);
        impl<D> Index<usize> for Forward<&D>
        where
            D: Dimension,
        {
            type Output = usize;
            #[inline]
            fn index(&self, index: usize) -> &usize {
                &self.0[index]
            }
        }
        impl<D> Index<usize> for Forward<&mut D>
        where
            D: Dimension,
        {
            type Output = usize;
            #[inline]
            fn index(&self, index: usize) -> &usize {
                &self.0[index]
            }
        }
        impl<D> IndexMut<usize> for Forward<&mut D>
        where
            D: Dimension,
        {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut usize {
                &mut self.0[index]
            }
        }
        impl<D> Index<usize> for Reverse<&D>
        where
            D: Dimension,
        {
            type Output = usize;
            #[inline]
            fn index(&self, index: usize) -> &usize {
                &self.0[self.len() - index - 1]
            }
        }
        impl<D> Index<usize> for Reverse<&mut D>
        where
            D: Dimension,
        {
            type Output = usize;
            #[inline]
            fn index(&self, index: usize) -> &usize {
                &self.0[self.len() - index - 1]
            }
        }
        impl<D> IndexMut<usize> for Reverse<&mut D>
        where
            D: Dimension,
        {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut usize {
                let len = self.len();
                &mut self.0[len - index - 1]
            }
        }
        #[doc = " Indexable sequence with length"]
        pub(in crate::dimension) trait Sequence: Index<usize> {
            fn len(&self) -> usize;
        }
        #[doc = " Indexable sequence with length (mut)"]
        pub(in crate::dimension) trait SequenceMut:
            Sequence + IndexMut<usize>
        {
        }
        impl<D> Sequence for Forward<&D>
        where
            D: Dimension,
        {
            #[inline]
            fn len(&self) -> usize {
                self.0.ndim()
            }
        }
        impl<D> Sequence for Forward<&mut D>
        where
            D: Dimension,
        {
            #[inline]
            fn len(&self) -> usize {
                self.0.ndim()
            }
        }
        impl<D> SequenceMut for Forward<&mut D> where D: Dimension {}
        impl<D> Sequence for Reverse<&D>
        where
            D: Dimension,
        {
            #[inline]
            fn len(&self) -> usize {
                self.0.ndim()
            }
        }
        impl<D> Sequence for Reverse<&mut D>
        where
            D: Dimension,
        {
            #[inline]
            fn len(&self) -> usize {
                self.0.ndim()
            }
        }
        impl<D> SequenceMut for Reverse<&mut D> where D: Dimension {}
    }
    #[doc = " Calculate offset from `Ix` stride converting sign properly"]
    #[inline(always)]
    pub fn stride_offset(n: Ix, stride: Ix) -> isize {
        (n as isize) * ((stride as Ixs) as isize)
    }
    #[doc = " Check whether the given `dim` and `stride` lead to overlapping indices"]
    #[doc = ""]
    #[doc = " There is overlap if, when iterating through the dimensions in order of"]
    #[doc = " increasing stride, the current stride is less than or equal to the maximum"]
    #[doc = " possible offset along the preceding axes. (Axes of length ≤1 are ignored.)"]
    pub fn dim_stride_overlap<D: Dimension>(dim: &D, strides: &D) -> bool {
        let order = strides._fastest_varying_stride_order();
        let mut sum_prev_offsets = 0;
        for &index in order.slice() {
            let d = dim[index];
            let s = (strides[index] as isize).abs();
            match d {
                0 => return false,
                1 => {}
                _ => {
                    if s <= sum_prev_offsets {
                        return true;
                    }
                    sum_prev_offsets += (d - 1) as isize * s;
                }
            }
        }
        false
    }
    #[doc = " Returns the `size` of the `dim`, checking that the product of non-zero axis"]
    #[doc = " lengths does not exceed `isize::MAX`."]
    #[doc = ""]
    #[doc = " If `size_of_checked_shape(dim)` returns `Ok(size)`, the data buffer is a"]
    #[doc = " slice or `Vec` of length `size`, and `strides` are created with"]
    #[doc = " `self.default_strides()` or `self.fortran_strides()`, then the invariants"]
    #[doc = " are met to construct an array from the data buffer, `dim`, and `strides`."]
    #[doc = " (The data buffer being a slice or `Vec` guarantees that it contains no more"]
    #[doc = " than `isize::MAX` bytes.)"]
    pub fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ShapeError> {
        let size_nonzero = dim
            .slice()
            .iter()
            .filter(|&&d| d != 0)
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
        if size_nonzero > ::std::isize::MAX as usize {
            Err(from_kind(ErrorKind::Overflow))
        } else {
            Ok(dim.size())
        }
    }
    #[doc = " Checks whether the given data and dimension meet the invariants of the"]
    #[doc = " `ArrayBase` type, assuming the strides are created using"]
    #[doc = " `dim.default_strides()` or `dim.fortran_strides()`."]
    #[doc = ""]
    #[doc = " To meet the invariants,"]
    #[doc = ""]
    #[doc = " 1. The product of non-zero axis lengths must not exceed `isize::MAX`."]
    #[doc = ""]
    #[doc = " 2. The result of `dim.size()` (assuming no overflow) must be less than or"]
    #[doc = "    equal to the length of the slice."]
    #[doc = ""]
    #[doc = "    (Since `dim.default_strides()` and `dim.fortran_strides()` always return"]
    #[doc = "    contiguous strides for non-empty arrays, this ensures that for non-empty"]
    #[doc = "    arrays the difference between the least address and greatest address"]
    #[doc = "    accessible by moving along all axes is < the length of the slice. Since"]
    #[doc = "    `dim.default_strides()` and `dim.fortran_strides()` always return all"]
    #[doc = "    zero strides for empty arrays, this ensures that for empty arrays the"]
    #[doc = "    difference between the least address and greatest address accessible by"]
    #[doc = "    moving along all axes is ≤ the length of the slice.)"]
    #[doc = ""]
    #[doc = " Note that since slices cannot contain more than `isize::MAX` bytes,"]
    #[doc = " conditions 1 and 2 are sufficient to guarantee that the offset in units of"]
    #[doc = " `A` and in units of bytes between the least address and greatest address"]
    #[doc = " accessible by moving along all axes does not exceed `isize::MAX`."]
    pub(crate) fn can_index_slice_with_strides<A, D: Dimension>(
        data: &[A],
        dim: &D,
        strides: &Strides<D>,
    ) -> Result<(), ShapeError> {
        if let Strides::Custom(strides) = strides {
            can_index_slice(data, dim, strides)
        } else {
            can_index_slice_not_custom(data.len(), dim)
        }
    }
    pub(crate) fn can_index_slice_not_custom<D: Dimension>(
        data_len: usize,
        dim: &D,
    ) -> Result<(), ShapeError> {
        let len = size_of_shape_checked(dim)?;
        if len > data_len {
            return Err(from_kind(ErrorKind::OutOfBounds));
        }
        Ok(())
    }
    #[doc = " Returns the absolute difference in units of `A` between least and greatest"]
    #[doc = " address accessible by moving along all axes."]
    #[doc = ""]
    #[doc = " Returns `Ok` only if"]
    #[doc = ""]
    #[doc = " 1. The ndim of `dim` and `strides` is the same."]
    #[doc = ""]
    #[doc = " 2. The absolute difference in units of `A` and in units of bytes between"]
    #[doc = "    the least address and greatest address accessible by moving along all axes"]
    #[doc = "    does not exceed `isize::MAX`."]
    #[doc = ""]
    #[doc = " 3. The product of non-zero axis lengths does not exceed `isize::MAX`. (This"]
    #[doc = "    also implies that the length of any individual axis does not exceed"]
    #[doc = "    `isize::MAX`.)"]
    pub fn max_abs_offset_check_overflow<A, D>(dim: &D, strides: &D) -> Result<usize, ShapeError>
    where
        D: Dimension,
    {
        max_abs_offset_check_overflow_impl(mem::size_of::<A>(), dim, strides)
    }
    fn max_abs_offset_check_overflow_impl<D>(
        elem_size: usize,
        dim: &D,
        strides: &D,
    ) -> Result<usize, ShapeError>
    where
        D: Dimension,
    {
        if dim.ndim() != strides.ndim() {
            return Err(from_kind(ErrorKind::IncompatibleLayout));
        }
        let _ = size_of_shape_checked(dim)?;
        let max_offset: usize = izip!(dim.slice(), strides.slice())
            .try_fold(0usize, |acc, (&d, &s)| {
                let s = s as isize;
                let off = d.saturating_sub(1).checked_mul(s.unsigned_abs())?;
                acc.checked_add(off)
            })
            .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
        if max_offset > isize::MAX as usize {
            return Err(from_kind(ErrorKind::Overflow));
        }
        let max_offset_bytes = max_offset
            .checked_mul(elem_size)
            .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
        if max_offset_bytes > isize::MAX as usize {
            return Err(from_kind(ErrorKind::Overflow));
        }
        Ok(max_offset)
    }
    #[doc = " Checks whether the given data, dimension, and strides meet the invariants"]
    #[doc = " of the `ArrayBase` type (except for checking ownership of the data)."]
    #[doc = ""]
    #[doc = " To meet the invariants,"]
    #[doc = ""]
    #[doc = " 1. The ndim of `dim` and `strides` must be the same."]
    #[doc = ""]
    #[doc = " 2. The product of non-zero axis lengths must not exceed `isize::MAX`."]
    #[doc = ""]
    #[doc = " 3. If the array will be empty (any axes are zero-length), the difference"]
    #[doc = "    between the least address and greatest address accessible by moving"]
    #[doc = "    along all axes must be ≤ `data.len()`. (It's fine in this case to move"]
    #[doc = "    one byte past the end of the slice since the pointers will be offset but"]
    #[doc = "    never dereferenced.)"]
    #[doc = ""]
    #[doc = "    If the array will not be empty, the difference between the least address"]
    #[doc = "    and greatest address accessible by moving along all axes must be <"]
    #[doc = "    `data.len()`. This and #3 ensure that all dereferenceable pointers point"]
    #[doc = "    to elements within the slice."]
    #[doc = ""]
    #[doc = " 4. The strides must not allow any element to be referenced by two different"]
    #[doc = "    indices."]
    #[doc = ""]
    #[doc = " Note that since slices cannot contain more than `isize::MAX` bytes,"]
    #[doc = " condition 4 is sufficient to guarantee that the absolute difference in"]
    #[doc = " units of `A` and in units of bytes between the least address and greatest"]
    #[doc = " address accessible by moving along all axes does not exceed `isize::MAX`."]
    #[doc = ""]
    #[doc = " Warning: This function is sufficient to check the invariants of ArrayBase"]
    #[doc = " only if the pointer to the first element of the array is chosen such that"]
    #[doc = " the element with the smallest memory address is at the start of the"]
    #[doc = " allocation. (In other words, the pointer to the first element of the array"]
    #[doc = " must be computed using `offset_from_low_addr_ptr_to_logical_ptr` so that"]
    #[doc = " negative strides are correctly handled.)"]
    pub(crate) fn can_index_slice<A, D: Dimension>(
        data: &[A],
        dim: &D,
        strides: &D,
    ) -> Result<(), ShapeError> {
        let max_offset = max_abs_offset_check_overflow::<A, _>(dim, strides)?;
        can_index_slice_impl(max_offset, data.len(), dim, strides)
    }
    fn can_index_slice_impl<D: Dimension>(
        max_offset: usize,
        data_len: usize,
        dim: &D,
        strides: &D,
    ) -> Result<(), ShapeError> {
        let is_empty = dim.slice().iter().any(|&d| d == 0);
        if is_empty && max_offset > data_len {
            return Err(from_kind(ErrorKind::OutOfBounds));
        }
        if !is_empty && max_offset >= data_len {
            return Err(from_kind(ErrorKind::OutOfBounds));
        }
        if !is_empty && dim_stride_overlap(dim, strides) {
            return Err(from_kind(ErrorKind::Unsupported));
        }
        Ok(())
    }
    #[doc = " Stride offset checked general version (slices)"]
    #[inline]
    pub fn stride_offset_checked(dim: &[Ix], strides: &[Ix], index: &[Ix]) -> Option<isize> {
        if index.len() != dim.len() {
            return None;
        }
        let mut offset = 0;
        for (&d, &i, &s) in izip!(dim, index, strides) {
            if i >= d {
                return None;
            }
            offset += stride_offset(i, s);
        }
        Some(offset)
    }
    #[doc = " Checks if strides are non-negative."]
    pub fn strides_non_negative<D>(strides: &D) -> Result<(), ShapeError>
    where
        D: Dimension,
    {
        for &stride in strides.slice() {
            if (stride as isize) < 0 {
                return Err(from_kind(ErrorKind::Unsupported));
            }
        }
        Ok(())
    }
    #[doc = " Implementation-specific extensions to `Dimension`"]
    pub trait DimensionExt {
        #[doc = " Get the dimension at `axis`."]
        #[doc = ""]
        #[doc = " *Panics* if `axis` is out of bounds."]
        fn axis(&self, axis: Axis) -> Ix;
        #[doc = " Set the dimension at `axis`."]
        #[doc = ""]
        #[doc = " *Panics* if `axis` is out of bounds."]
        fn set_axis(&mut self, axis: Axis, value: Ix);
    }
    impl<D> DimensionExt for D
    where
        D: Dimension,
    {
        #[inline]
        fn axis(&self, axis: Axis) -> Ix {
            self[axis.index()]
        }
        #[inline]
        fn set_axis(&mut self, axis: Axis, value: Ix) {
            self[axis.index()] = value;
        }
    }
    #[doc = " Collapse axis `axis` and shift so that only subarray `index` is"]
    #[doc = " available."]
    #[doc = ""]
    #[doc = " **Panics** if `index` is larger than the size of the axis"]
    pub fn do_collapse_axis<D: Dimension>(
        dims: &mut D,
        strides: &D,
        axis: usize,
        index: usize,
    ) -> isize {
        let dim = dims.slice()[axis];
        let stride = strides.slice()[axis];
        ndassert!(
            index < dim,
            "collapse_axis: Index {} must be less than axis length {} for \
         array with shape {:?}",
            index,
            dim,
            *dims
        );
        dims.slice_mut()[axis] = 1;
        stride_offset(index, stride)
    }
    #[doc = " Compute the equivalent unsigned index given the axis length and signed index."]
    #[inline]
    pub fn abs_index(len: Ix, index: Ixs) -> Ix {
        if index < 0 {
            len - (-index as Ix)
        } else {
            index as Ix
        }
    }
    #[doc = " Determines nonnegative start and end indices, and performs sanity checks."]
    #[doc = ""]
    #[doc = " The return value is (start, end, step)."]
    #[doc = ""]
    #[doc = " **Panics** if stride is 0 or if any index is out of bounds."]
    fn to_abs_slice(axis_len: usize, slice: Slice) -> (usize, usize, isize) {
        let Slice { start, end, step } = slice;
        let start = abs_index(axis_len, start);
        let mut end = abs_index(axis_len, end.unwrap_or(axis_len as isize));
        if end < start {
            end = start;
        }
        ndassert!(
            start <= axis_len,
            "Slice begin {} is past end of axis of length {}",
            start,
            axis_len,
        );
        ndassert!(
            end <= axis_len,
            "Slice end {} is past end of axis of length {}",
            end,
            axis_len,
        );
        ndassert!(step != 0, "Slice stride must not be zero");
        (start, end, step)
    }
    #[doc = " Returns the offset from the lowest-address element to the logically first"]
    #[doc = " element."]
    pub fn offset_from_low_addr_ptr_to_logical_ptr<D: Dimension>(dim: &D, strides: &D) -> usize {
        let offset = izip!(dim.slice(), strides.slice()).fold(0, |_offset, (&d, &s)| {
            let s = s as isize;
            if s < 0 && d > 1 {
                _offset - s * (d as isize - 1)
            } else {
                _offset
            }
        });
        debug_assert!(offset >= 0);
        offset as usize
    }
    #[doc = " Modify dimension, stride and return data pointer offset"]
    #[doc = ""]
    #[doc = " **Panics** if stride is 0 or if any index is out of bounds."]
    pub fn do_slice(dim: &mut usize, stride: &mut usize, slice: Slice) -> isize {
        let (start, end, step) = to_abs_slice(*dim, slice);
        let m = end - start;
        let s = (*stride) as isize;
        let offset = if m == 0 {
            0
        } else if step < 0 {
            stride_offset(end - 1, *stride)
        } else {
            stride_offset(start, *stride)
        };
        let abs_step = step.unsigned_abs();
        *dim = if abs_step == 1 {
            m
        } else {
            let d = m / abs_step;
            let r = m % abs_step;
            d + if r > 0 { 1 } else { 0 }
        };
        *stride = if *dim <= 1 { 0 } else { (s * step) as usize };
        offset
    }
    #[doc = " Solves `a * x + b * y = gcd(a, b)` for `x`, `y`, and `gcd(a, b)`."]
    #[doc = ""]
    #[doc = " Returns `(g, (x, y))`, where `g` is `gcd(a, b)`, and `g` is always"]
    #[doc = " nonnegative."]
    #[doc = ""]
    #[doc = " See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm"]
    fn extended_gcd(a: isize, b: isize) -> (isize, (isize, isize)) {
        if a == 0 {
            (b.abs(), (0, b.signum()))
        } else if b == 0 {
            (a.abs(), (a.signum(), 0))
        } else {
            let mut r = (a, b);
            let mut s = (1, 0);
            let mut t = (0, 1);
            while r.1 != 0 {
                let q = r.0 / r.1;
                r = (r.1, r.0 - q * r.1);
                s = (s.1, s.0 - q * s.1);
                t = (t.1, t.0 - q * t.1);
            }
            if r.0 > 0 {
                (r.0, (s.0, t.0))
            } else {
                (-r.0, (-s.0, -t.0))
            }
        }
    }
    #[doc = " Solves `a * x + b * y = c` for `x` where `a`, `b`, `c`, `x`, and `y` are"]
    #[doc = " integers."]
    #[doc = ""]
    #[doc = " If the return value is `Some((x0, xd))`, there is a solution. `xd` is"]
    #[doc = " always positive. Solutions `x` are given by `x0 + xd * t` where `t` is any"]
    #[doc = " integer. The value of `y` for any `x` is then `y = (c - a * x) / b`."]
    #[doc = ""]
    #[doc = " If the return value is `None`, no solutions exist."]
    #[doc = ""]
    #[doc = " **Note** `a` and `b` must be nonzero."]
    #[doc = ""]
    #[doc = " See https://en.wikipedia.org/wiki/Diophantine_equation#One_equation"]
    #[doc = " and https://math.stackexchange.com/questions/1656120#1656138"]
    fn solve_linear_diophantine_eq(a: isize, b: isize, c: isize) -> Option<(isize, isize)> {
        debug_assert_ne!(a, 0);
        debug_assert_ne!(b, 0);
        let (g, (u, _)) = extended_gcd(a, b);
        if c % g == 0 {
            Some((c / g * u, (b / g).abs()))
        } else {
            None
        }
    }
    #[doc = " Returns `true` if two (finite length) arithmetic sequences intersect."]
    #[doc = ""]
    #[doc = " `min*` and `max*` are the (inclusive) bounds of the sequences, and they"]
    #[doc = " must be elements in the sequences. `step*` are the steps between"]
    #[doc = " consecutive elements (the sign is irrelevant)."]
    #[doc = ""]
    #[doc = " **Note** `step1` and `step2` must be nonzero."]
    fn arith_seq_intersect(
        (min1, max1, step1): (isize, isize, isize),
        (min2, max2, step2): (isize, isize, isize),
    ) -> bool {
        debug_assert!(max1 >= min1);
        debug_assert!(max2 >= min2);
        debug_assert_eq!((max1 - min1) % step1, 0);
        debug_assert_eq!((max2 - min2) % step2, 0);
        if min1 > max2 || min2 > max1 {
            false
        } else {
            let step1 = step1.abs();
            let step2 = step2.abs();
            if let Some((x0, xd)) = solve_linear_diophantine_eq(-step1, step2, min1 - min2) {
                let min = ::std::cmp::max(min1, min2);
                let max = ::std::cmp::min(max1, max2);
                min1 + step1 * (x0 - xd * div_floor(min - min1 - step1 * x0, -step1 * xd)) <= max
                    || min1 + step1 * (x0 + xd * div_floor(max - min1 - step1 * x0, step1 * xd))
                        >= min
            } else {
                false
            }
        }
    }
    #[doc = " Returns the minimum and maximum values of the indices (inclusive)."]
    #[doc = ""]
    #[doc = " If the slice is empty, then returns `None`, otherwise returns `Some((min, max))`."]
    fn slice_min_max(axis_len: usize, slice: Slice) -> Option<(usize, usize)> {
        let (start, end, step) = to_abs_slice(axis_len, slice);
        if start == end {
            None
        } else if step > 0 {
            Some((start, end - 1 - (end - start - 1) % (step as usize)))
        } else {
            Some((start + (end - start - 1) % (-step as usize), end - 1))
        }
    }
    #[doc = " Returns `true` iff the slices intersect."]
    pub fn slices_intersect<D: Dimension>(
        dim: &D,
        indices1: impl SliceArg<D>,
        indices2: impl SliceArg<D>,
    ) -> bool {
        debug_assert_eq!(indices1.in_ndim(), indices2.in_ndim());
        for (&axis_len, &si1, &si2) in izip!(
            dim.slice(),
            indices1.as_ref().iter().filter(|si| !si.is_new_axis()),
            indices2.as_ref().iter().filter(|si| !si.is_new_axis()),
        ) {
            match (si1, si2) {
                (
                    SliceInfoElem::Slice {
                        start: start1,
                        end: end1,
                        step: step1,
                    },
                    SliceInfoElem::Slice {
                        start: start2,
                        end: end2,
                        step: step2,
                    },
                ) => {
                    let (min1, max1) =
                        match slice_min_max(axis_len, Slice::new(start1, end1, step1)) {
                            Some(m) => m,
                            None => return false,
                        };
                    let (min2, max2) =
                        match slice_min_max(axis_len, Slice::new(start2, end2, step2)) {
                            Some(m) => m,
                            None => return false,
                        };
                    if !arith_seq_intersect(
                        (min1 as isize, max1 as isize, step1),
                        (min2 as isize, max2 as isize, step2),
                    ) {
                        return false;
                    }
                }
                (SliceInfoElem::Slice { start, end, step }, SliceInfoElem::Index(ind))
                | (SliceInfoElem::Index(ind), SliceInfoElem::Slice { start, end, step }) => {
                    let ind = abs_index(axis_len, ind);
                    let (min, max) = match slice_min_max(axis_len, Slice::new(start, end, step)) {
                        Some(m) => m,
                        None => return false,
                    };
                    if ind < min || ind > max || (ind - min) % step.unsigned_abs() != 0 {
                        return false;
                    }
                }
                (SliceInfoElem::Index(ind1), SliceInfoElem::Index(ind2)) => {
                    let ind1 = abs_index(axis_len, ind1);
                    let ind2 = abs_index(axis_len, ind2);
                    if ind1 != ind2 {
                        return false;
                    }
                }
                (SliceInfoElem::NewAxis, _) | (_, SliceInfoElem::NewAxis) => unreachable!(),
            }
        }
        true
    }
    pub(crate) fn is_layout_c<D: Dimension>(dim: &D, strides: &D) -> bool {
        if let Some(1) = D::NDIM {
            return strides[0] == 1 || dim[0] <= 1;
        }
        for &d in dim.slice() {
            if d == 0 {
                return true;
            }
        }
        let mut contig_stride = 1_isize;
        for (&dim, &s) in izip!(dim.slice().iter().rev(), strides.slice().iter().rev()) {
            if dim != 1 {
                let s = s as isize;
                if s != contig_stride {
                    return false;
                }
                contig_stride *= dim as isize;
            }
        }
        true
    }
    pub(crate) fn is_layout_f<D: Dimension>(dim: &D, strides: &D) -> bool {
        if let Some(1) = D::NDIM {
            return strides[0] == 1 || dim[0] <= 1;
        }
        for &d in dim.slice() {
            if d == 0 {
                return true;
            }
        }
        let mut contig_stride = 1_isize;
        for (&dim, &s) in izip!(dim.slice(), strides.slice()) {
            if dim != 1 {
                let s = s as isize;
                if s != contig_stride {
                    return false;
                }
                contig_stride *= dim as isize;
            }
        }
        true
    }
    pub fn merge_axes<D>(dim: &mut D, strides: &mut D, take: Axis, into: Axis) -> bool
    where
        D: Dimension,
    {
        let into_len = dim.axis(into);
        let into_stride = strides.axis(into) as isize;
        let take_len = dim.axis(take);
        let take_stride = strides.axis(take) as isize;
        let merged_len = into_len * take_len;
        if take_len <= 1 {
            dim.set_axis(into, merged_len);
            dim.set_axis(take, if merged_len == 0 { 0 } else { 1 });
            true
        } else if into_len <= 1 {
            strides.set_axis(into, take_stride as usize);
            dim.set_axis(into, merged_len);
            dim.set_axis(take, if merged_len == 0 { 0 } else { 1 });
            true
        } else if take_stride == into_len as isize * into_stride {
            dim.set_axis(into, merged_len);
            dim.set_axis(take, 1);
            true
        } else {
            false
        }
    }
    #[doc = " Move the axis which has the smallest absolute stride and a length"]
    #[doc = " greater than one to be the last axis."]
    pub fn move_min_stride_axis_to_last<D>(dim: &mut D, strides: &mut D)
    where
        D: Dimension,
    {
        debug_assert_eq!(dim.ndim(), strides.ndim());
        match dim.ndim() {
            0 | 1 => {}
            2 => {
                if dim[1] <= 1
                    || dim[0] > 1 && (strides[0] as isize).abs() < (strides[1] as isize).abs()
                {
                    dim.slice_mut().swap(0, 1);
                    strides.slice_mut().swap(0, 1);
                }
            }
            n => {
                if let Some(min_stride_axis) = (0..n)
                    .filter(|&ax| dim[ax] > 1)
                    .min_by_key(|&ax| (strides[ax] as isize).abs())
                {
                    let last = n - 1;
                    dim.slice_mut().swap(last, min_stride_axis);
                    strides.slice_mut().swap(last, min_stride_axis);
                }
            }
        }
    }
}
pub use crate::layout::Layout;
pub use crate::zip::{FoldWhile, IntoNdProducer, NdProducer, Zip};
#[doc = " Implementation's prelude. Common types used everywhere."]
mod imp_prelude {
    pub use crate::dimension::DimensionExt;
    pub use crate::prelude::*;
    pub use crate::{
        CowRepr, Data, DataMut, DataOwned, DataShared, Ix, Ixs, RawData, RawDataMut, RawViewRepr,
        RemoveAxis, ViewRepr,
    };
}
pub mod prelude {
    #![doc = " ndarray prelude."]
    #![doc = ""]
    #![doc = " This module contains the most used types, type aliases, traits, functions,"]
    #![doc = " and macros that you can import easily as a group."]
    #![doc = ""]
    #![doc = " ```"]
    #![doc = " use ndarray::prelude::*;"]
    #![doc = ""]
    #![doc = " # let _ = arr0(1); // use the import"]
    #![doc = " ```"]
    #[doc(no_inline)]
    pub use crate::ShapeBuilder;
    #[doc(no_inline)]
    pub use crate::{
        ArcArray, Array, ArrayBase, ArrayView, ArrayViewMut, CowArray, RawArrayView,
        RawArrayViewMut,
    };
    #[doc(no_inline)]
    pub use crate::{Array0, Array1, Array2, Array3, Array4, Array5, Array6, ArrayD};
    #[doc(no_inline)]
    pub use crate::{
        ArrayView0, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayView5, ArrayView6,
        ArrayViewD,
    };
    #[doc(no_inline)]
    pub use crate::{
        ArrayViewMut0, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMut4, ArrayViewMut5,
        ArrayViewMut6, ArrayViewMutD,
    };
    #[doc(no_inline)]
    pub use crate::{Axis, Dim, Dimension};
    #[doc(no_inline)]
    pub use crate::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
}
#[doc = " Array index type"]
pub type Ix = usize;
#[doc = " Array index type (signed)"]
pub type Ixs = isize;
#[doc = " An *n*-dimensional array."]
#[doc = ""]
#[doc = " The array is a general container of elements."]
#[doc = " The array supports arithmetic operations by applying them elementwise, if the"]
#[doc = " elements are numeric, but it supports non-numeric elements too."]
#[doc = ""]
#[doc = " The arrays rarely grow or shrink, since those operations can be costly. On"]
#[doc = " the other hand there is a rich set of methods and operations for taking views,"]
#[doc = " slices, and making traversals over one or more arrays."]
#[doc = ""]
#[doc = " In *n*-dimensional we include for example 1-dimensional rows or columns,"]
#[doc = " 2-dimensional matrices, and higher dimensional arrays. If the array has *n*"]
#[doc = " dimensions, then an element is accessed by using that many indices."]
#[doc = ""]
#[doc = " The `ArrayBase<S, D>` is parameterized by `S` for the data container and"]
#[doc = " `D` for the dimensionality."]
#[doc = ""]
#[doc = " Type aliases [`Array`], [`ArcArray`], [`CowArray`], [`ArrayView`], and"]
#[doc = " [`ArrayViewMut`] refer to `ArrayBase` with different types for the data"]
#[doc = " container: arrays with different kinds of ownership or different kinds of array views."]
#[doc = ""]
#[doc = " ## Contents"]
#[doc = ""]
#[doc = " + [Array](#array)"]
#[doc = " + [ArcArray](#arcarray)"]
#[doc = " + [CowArray](#cowarray)"]
#[doc = " + [Array Views](#array-views)"]
#[doc = " + [Indexing and Dimension](#indexing-and-dimension)"]
#[doc = " + [Loops, Producers and Iterators](#loops-producers-and-iterators)"]
#[doc = " + [Slicing](#slicing)"]
#[doc = " + [Subviews](#subviews)"]
#[doc = " + [Arithmetic Operations](#arithmetic-operations)"]
#[doc = " + [Broadcasting](#broadcasting)"]
#[doc = " + [Conversions](#conversions)"]
#[doc = " + [Constructor Methods for Owned Arrays](#constructor-methods-for-owned-arrays)"]
#[doc = " + [Methods For All Array Types](#methods-for-all-array-types)"]
#[doc = " + [Methods For 1-D Arrays](#methods-for-1-d-arrays)"]
#[doc = " + [Methods For 2-D Arrays](#methods-for-2-d-arrays)"]
#[doc = " + [Methods for Dynamic-Dimensional Arrays](#methods-for-dynamic-dimensional-arrays)"]
#[doc = " + [Numerical Methods for Arrays](#numerical-methods-for-arrays)"]
#[doc = ""]
#[doc = " ## `Array`"]
#[doc = ""]
#[doc = " [`Array`] is an owned array that owns the underlying array"]
#[doc = " elements directly (just like a `Vec`) and it is the default way to create and"]
#[doc = " store n-dimensional data. `Array<A, D>` has two type parameters: `A` for"]
#[doc = " the element type, and `D` for the dimensionality. A particular"]
#[doc = " dimensionality's type alias like `Array3<A>` just has the type parameter"]
#[doc = " `A` for element type."]
#[doc = ""]
#[doc = " An example:"]
#[doc = ""]
#[doc = " ```"]
#[doc = " // Create a three-dimensional f64 array, initialized with zeros"]
#[doc = " use ndarray::Array3;"]
#[doc = " let mut temperature = Array3::<f64>::zeros((3, 4, 5));"]
#[doc = " // Increase the temperature in this location"]
#[doc = " temperature[[2, 2, 2]] += 0.5;"]
#[doc = " ```"]
#[doc = ""]
#[doc = " ## `ArcArray`"]
#[doc = ""]
#[doc = " [`ArcArray`] is an owned array with reference counted"]
#[doc = " data (shared ownership)."]
#[doc = " Sharing requires that it uses copy-on-write for mutable operations."]
#[doc = " Calling a method for mutating elements on `ArcArray`, for example"]
#[doc = " [`view_mut()`](Self::view_mut) or [`get_mut()`](Self::get_mut),"]
#[doc = " will break sharing and require a clone of the data (if it is not uniquely held)."]
#[doc = ""]
#[doc = " ## `CowArray`"]
#[doc = ""]
#[doc = " [`CowArray`] is analogous to [`std::borrow::Cow`]."]
#[doc = " It can represent either an immutable view or a uniquely owned array. If a"]
#[doc = " `CowArray` instance is the immutable view variant, then calling a method"]
#[doc = " for mutating elements in the array will cause it to be converted into the"]
#[doc = " owned variant (by cloning all the elements) before the modification is"]
#[doc = " performed."]
#[doc = ""]
#[doc = " ## Array Views"]
#[doc = ""]
#[doc = " [`ArrayView`] and [`ArrayViewMut`] are read-only and read-write array views"]
#[doc = " respectively. They use dimensionality, indexing, and almost all other"]
#[doc = " methods the same way as the other array types."]
#[doc = ""]
#[doc = " Methods for `ArrayBase` apply to array views too, when the trait bounds"]
#[doc = " allow."]
#[doc = ""]
#[doc = " Please see the documentation for the respective array view for an overview"]
#[doc = " of methods specific to array views: [`ArrayView`], [`ArrayViewMut`]."]
#[doc = ""]
#[doc = " A view is created from an array using [`.view()`](ArrayBase::view),"]
#[doc = " [`.view_mut()`](ArrayBase::view_mut), using"]
#[doc = " slicing ([`.slice()`](ArrayBase::slice), [`.slice_mut()`](ArrayBase::slice_mut)) or from one of"]
#[doc = " the many iterators that yield array views."]
#[doc = ""]
#[doc = " You can also create an array view from a regular slice of data not"]
#[doc = " allocated with `Array` — see array view methods or their `From` impls."]
#[doc = ""]
#[doc = " Note that all `ArrayBase` variants can change their view (slicing) of the"]
#[doc = " data freely, even when their data can’t be mutated."]
#[doc = ""]
#[doc = " ## Indexing and Dimension"]
#[doc = ""]
#[doc = " The dimensionality of the array determines the number of *axes*, for example"]
#[doc = " a 2D array has two axes. These are listed in “big endian” order, so that"]
#[doc = " the greatest dimension is listed first, the lowest dimension with the most"]
#[doc = " rapidly varying index is the last."]
#[doc = ""]
#[doc = " In a 2D array the index of each element is `[row, column]` as seen in this"]
#[doc = " 4 × 3 example:"]
#[doc = ""]
#[doc = " ```ignore"]
#[doc = " [[ [0, 0], [0, 1], [0, 2] ],  // row 0"]
#[doc = "  [ [1, 0], [1, 1], [1, 2] ],  // row 1"]
#[doc = "  [ [2, 0], [2, 1], [2, 2] ],  // row 2"]
#[doc = "  [ [3, 0], [3, 1], [3, 2] ]]  // row 3"]
#[doc = " //    \\       \\       \\"]
#[doc = " //   column 0  \\     column 2"]
#[doc = " //            column 1"]
#[doc = " ```"]
#[doc = ""]
#[doc = " The number of axes for an array is fixed by its `D` type parameter: `Ix1`"]
#[doc = " for a 1D array, `Ix2` for a 2D array etc. The dimension type `IxDyn` allows"]
#[doc = " a dynamic number of axes."]
#[doc = ""]
#[doc = " A fixed size array (`[usize; N]`) of the corresponding dimensionality is"]
#[doc = " used to index the `Array`, making the syntax `array[[` i, j,  ...`]]`"]
#[doc = ""]
#[doc = " ```"]
#[doc = " use ndarray::Array2;"]
#[doc = " let mut array = Array2::zeros((4, 3));"]
#[doc = " array[[1, 1]] = 7;"]
#[doc = " ```"]
#[doc = ""]
#[doc = " Important traits and types for dimension and indexing:"]
#[doc = ""]
#[doc = " - A [`struct@Dim`] value represents a dimensionality or index."]
#[doc = " - Trait [`Dimension`] is implemented by all"]
#[doc = " dimensionalities. It defines many operations for dimensions and indices."]
#[doc = " - Trait [`IntoDimension`] is used to convert into a"]
#[doc = " `Dim` value."]
#[doc = " - Trait [`ShapeBuilder`] is an extension of"]
#[doc = " `IntoDimension` and is used when constructing an array. A shape describes"]
#[doc = " not just the extent of each axis but also their strides."]
#[doc = " - Trait [`NdIndex`] is an extension of `Dimension` and is"]
#[doc = " for values that can be used with indexing syntax."]
#[doc = ""]
#[doc = ""]
#[doc = " The default memory order of an array is *row major* order (a.k.a “c” order),"]
#[doc = " where each row is contiguous in memory."]
#[doc = " A *column major* (a.k.a. “f” or fortran) memory order array has"]
#[doc = " columns (or, in general, the outermost axis) with contiguous elements."]
#[doc = ""]
#[doc = " The logical order of any array’s elements is the row major order"]
#[doc = " (the rightmost index is varying the fastest)."]
#[doc = " The iterators `.iter(), .iter_mut()` always adhere to this order, for example."]
#[doc = ""]
#[doc = " ## Loops, Producers and Iterators"]
#[doc = ""]
#[doc = " Using [`Zip`] is the most general way to apply a procedure"]
#[doc = " across one or several arrays or *producers*."]
#[doc = ""]
#[doc = " [`NdProducer`] is like an iterable but for"]
#[doc = " multidimensional data. All producers have dimensions and axes, like an"]
#[doc = " array view, and they can be split and used with parallelization using `Zip`."]
#[doc = ""]
#[doc = " For example, `ArrayView<A, D>` is a producer, it has the same dimensions"]
#[doc = " as the array view and for each iteration it produces a reference to"]
#[doc = " the array element (`&A` in this case)."]
#[doc = ""]
#[doc = " Another example, if we have a 10 × 10 array and use `.exact_chunks((2, 2))`"]
#[doc = " we get a producer of chunks which has the dimensions 5 × 5 (because"]
#[doc = " there are *10 / 2 = 5* chunks in either direction). The 5 × 5 chunks producer"]
#[doc = " can be paired with any other producers of the same dimension with `Zip`, for"]
#[doc = " example 5 × 5 arrays."]
#[doc = ""]
#[doc = " ### `.iter()` and `.iter_mut()`"]
#[doc = ""]
#[doc = " These are the element iterators of arrays and they produce an element"]
#[doc = " sequence in the logical order of the array, that means that the elements"]
#[doc = " will be visited in the sequence that corresponds to increasing the"]
#[doc = " last index first: *0, ..., 0,  0*; *0, ..., 0, 1*; *0, ...0, 2* and so on."]
#[doc = ""]
#[doc = " ### `.outer_iter()` and `.axis_iter()`"]
#[doc = ""]
#[doc = " These iterators produce array views of one smaller dimension."]
#[doc = ""]
#[doc = " For example, for a 2D array, `.outer_iter()` will produce the 1D rows."]
#[doc = " For a 3D array, `.outer_iter()` produces 2D subviews."]
#[doc = ""]
#[doc = " `.axis_iter()` is like `outer_iter()` but allows you to pick which"]
#[doc = " axis to traverse."]
#[doc = ""]
#[doc = " The `outer_iter` and `axis_iter` are one dimensional producers."]
#[doc = ""]
#[doc = " ## `.rows()`, `.columns()` and `.lanes()`"]
#[doc = ""]
#[doc = " [`.rows()`][gr] is a producer (and iterable) of all rows in an array."]
#[doc = ""]
#[doc = " ```"]
#[doc = " use ndarray::Array;"]
#[doc = ""]
#[doc = " // 1. Loop over the rows of a 2D array"]
#[doc = " let mut a = Array::zeros((10, 10));"]
#[doc = " for mut row in a.rows_mut() {"]
#[doc = "     row.fill(1.);"]
#[doc = " }"]
#[doc = ""]
#[doc = " // 2. Use Zip to pair each row in 2D `a` with elements in 1D `b`"]
#[doc = " use ndarray::Zip;"]
#[doc = " let mut b = Array::zeros(a.nrows());"]
#[doc = ""]
#[doc = " Zip::from(a.rows())"]
#[doc = "     .and(&mut b)"]
#[doc = "     .for_each(|a_row, b_elt| {"]
#[doc = "         *b_elt = a_row[a.ncols() - 1] - a_row[0];"]
#[doc = "     });"]
#[doc = " ```"]
#[doc = ""]
#[doc = " The *lanes* of an array are 1D segments along an axis and when pointed"]
#[doc = " along the last axis they are *rows*, when pointed along the first axis"]
#[doc = " they are *columns*."]
#[doc = ""]
#[doc = " A *m* × *n* array has *m* rows each of length *n* and conversely"]
#[doc = " *n* columns each of length *m*."]
#[doc = ""]
#[doc = " To generalize this, we say that an array of dimension *a* × *m* × *n*"]
#[doc = " has *a m* rows. It's composed of *a* times the previous array, so it"]
#[doc = " has *a* times as many rows."]
#[doc = ""]
#[doc = " All methods: [`.rows()`][gr], [`.rows_mut()`][grm],"]
#[doc = " [`.columns()`][gc], [`.columns_mut()`][gcm],"]
#[doc = " [`.lanes(axis)`][l], [`.lanes_mut(axis)`][lm]."]
#[doc = ""]
#[doc = " [gr]: Self::rows"]
#[doc = " [grm]: Self::rows_mut"]
#[doc = " [gc]: Self::columns"]
#[doc = " [gcm]: Self::columns_mut"]
#[doc = " [l]: Self::lanes"]
#[doc = " [lm]: Self::lanes_mut"]
#[doc = ""]
#[doc = " Yes, for 2D arrays `.rows()` and `.outer_iter()` have about the same"]
#[doc = " effect:"]
#[doc = ""]
#[doc = "  + `rows()` is a producer with *n* - 1 dimensions of 1 dimensional items"]
#[doc = "  + `outer_iter()` is a producer with 1 dimension of *n* - 1 dimensional items"]
#[doc = ""]
#[doc = " ## Slicing"]
#[doc = ""]
#[doc = " You can use slicing to create a view of a subset of the data in"]
#[doc = " the array. Slicing methods include [`.slice()`], [`.slice_mut()`],"]
#[doc = " [`.slice_move()`], and [`.slice_collapse()`]."]
#[doc = ""]
#[doc = " The slicing argument can be passed using the macro [`s![]`](s!),"]
#[doc = " which will be used in all examples. (The explicit form is an instance of"]
#[doc = " [`SliceInfo`] or another type which implements [`SliceArg`]; see their docs"]
#[doc = " for more information.)"]
#[doc = ""]
#[doc = " If a range is used, the axis is preserved. If an index is used, that index"]
#[doc = " is selected and the axis is removed; this selects a subview. See"]
#[doc = " [*Subviews*](#subviews) for more information about subviews. If a"]
#[doc = " [`NewAxis`] instance is used, a new axis is inserted. Note that"]
#[doc = " [`.slice_collapse()`] panics on `NewAxis` elements and behaves like"]
#[doc = " [`.collapse_axis()`] by preserving the number of dimensions."]
#[doc = ""]
#[doc = " [`.slice()`]: Self::slice"]
#[doc = " [`.slice_mut()`]: Self::slice_mut"]
#[doc = " [`.slice_move()`]: Self::slice_move"]
#[doc = " [`.slice_collapse()`]: Self::slice_collapse"]
#[doc = ""]
#[doc = " When slicing arrays with generic dimensionality, creating an instance of"]
#[doc = " [`SliceInfo`] to pass to the multi-axis slicing methods like [`.slice()`]"]
#[doc = " is awkward. In these cases, it's usually more convenient to use"]
#[doc = " [`.slice_each_axis()`]/[`.slice_each_axis_mut()`]/[`.slice_each_axis_inplace()`]"]
#[doc = " or to create a view and then slice individual axes of the view using"]
#[doc = " methods such as [`.slice_axis_inplace()`] and [`.collapse_axis()`]."]
#[doc = ""]
#[doc = " [`.slice_each_axis()`]: Self::slice_each_axis"]
#[doc = " [`.slice_each_axis_mut()`]: Self::slice_each_axis_mut"]
#[doc = " [`.slice_each_axis_inplace()`]: Self::slice_each_axis_inplace"]
#[doc = " [`.slice_axis_inplace()`]: Self::slice_axis_inplace"]
#[doc = " [`.collapse_axis()`]: Self::collapse_axis"]
#[doc = ""]
#[doc = " It's possible to take multiple simultaneous *mutable* slices with"]
#[doc = " [`.multi_slice_mut()`] or (for [`ArrayViewMut`] only)"]
#[doc = " [`.multi_slice_move()`]."]
#[doc = ""]
#[doc = " [`.multi_slice_mut()`]: Self::multi_slice_mut"]
#[doc = " [`.multi_slice_move()`]: ArrayViewMut#method.multi_slice_move"]
#[doc = ""]
#[doc = " ```"]
#[doc = " use ndarray::{arr2, arr3, s, ArrayBase, DataMut, Dimension, NewAxis, Slice};"]
#[doc = ""]
#[doc = " // 2 submatrices of 2 rows with 3 elements per row, means a shape of `[2, 2, 3]`."]
#[doc = ""]
#[doc = " let a = arr3(&[[[ 1,  2,  3],     // -- 2 rows  \\_"]
#[doc = "                 [ 4,  5,  6]],    // --         /"]
#[doc = "                [[ 7,  8,  9],     //            \\_ 2 submatrices"]
#[doc = "                 [10, 11, 12]]]);  //            /"]
#[doc = " //  3 columns ..../.../.../"]
#[doc = ""]
#[doc = " assert_eq!(a.shape(), &[2, 2, 3]);"]
#[doc = ""]
#[doc = " // Let’s create a slice with"]
#[doc = " //"]
#[doc = " // - Both of the submatrices of the greatest dimension: `..`"]
#[doc = " // - Only the first row in each submatrix: `0..1`"]
#[doc = " // - Every element in each row: `..`"]
#[doc = ""]
#[doc = " let b = a.slice(s![.., 0..1, ..]);"]
#[doc = " let c = arr3(&[[[ 1,  2,  3]],"]
#[doc = "                [[ 7,  8,  9]]]);"]
#[doc = " assert_eq!(b, c);"]
#[doc = " assert_eq!(b.shape(), &[2, 1, 3]);"]
#[doc = ""]
#[doc = " // Let’s create a slice with"]
#[doc = " //"]
#[doc = " // - Both submatrices of the greatest dimension: `..`"]
#[doc = " // - The last row in each submatrix: `-1..`"]
#[doc = " // - Row elements in reverse order: `..;-1`"]
#[doc = " let d = a.slice(s![.., -1.., ..;-1]);"]
#[doc = " let e = arr3(&[[[ 6,  5,  4]],"]
#[doc = "                [[12, 11, 10]]]);"]
#[doc = " assert_eq!(d, e);"]
#[doc = " assert_eq!(d.shape(), &[2, 1, 3]);"]
#[doc = ""]
#[doc = " // Let’s create a slice while selecting a subview and inserting a new axis with"]
#[doc = " //"]
#[doc = " // - Both submatrices of the greatest dimension: `..`"]
#[doc = " // - The last row in each submatrix, removing that axis: `-1`"]
#[doc = " // - Row elements in reverse order: `..;-1`"]
#[doc = " // - A new axis at the end."]
#[doc = " let f = a.slice(s![.., -1, ..;-1, NewAxis]);"]
#[doc = " let g = arr3(&[[ [6],  [5],  [4]],"]
#[doc = "                [[12], [11], [10]]]);"]
#[doc = " assert_eq!(f, g);"]
#[doc = " assert_eq!(f.shape(), &[2, 3, 1]);"]
#[doc = ""]
#[doc = " // Let's take two disjoint, mutable slices of a matrix with"]
#[doc = " //"]
#[doc = " // - One containing all the even-index columns in the matrix"]
#[doc = " // - One containing all the odd-index columns in the matrix"]
#[doc = " let mut h = arr2(&[[0, 1, 2, 3],"]
#[doc = "                    [4, 5, 6, 7]]);"]
#[doc = " let (s0, s1) = h.multi_slice_mut((s![.., ..;2], s![.., 1..;2]));"]
#[doc = " let i = arr2(&[[0, 2],"]
#[doc = "                [4, 6]]);"]
#[doc = " let j = arr2(&[[1, 3],"]
#[doc = "                [5, 7]]);"]
#[doc = " assert_eq!(s0, i);"]
#[doc = " assert_eq!(s1, j);"]
#[doc = ""]
#[doc = " // Generic function which assigns the specified value to the elements which"]
#[doc = " // have indices in the lower half along all axes."]
#[doc = " fn fill_lower<S, D>(arr: &mut ArrayBase<S, D>, x: S::Elem)"]
#[doc = " where"]
#[doc = "     S: DataMut,"]
#[doc = "     S::Elem: Clone,"]
#[doc = "     D: Dimension,"]
#[doc = " {"]
#[doc = "     arr.slice_each_axis_mut(|ax| Slice::from(0..ax.len / 2)).fill(x);"]
#[doc = " }"]
#[doc = " fill_lower(&mut h, 9);"]
#[doc = " let k = arr2(&[[9, 9, 2, 3],"]
#[doc = "                [4, 5, 6, 7]]);"]
#[doc = " assert_eq!(h, k);"]
#[doc = " ```"]
#[doc = ""]
#[doc = " ## Subviews"]
#[doc = ""]
#[doc = " Subview methods allow you to restrict the array view while removing one"]
#[doc = " axis from the array. Methods for selecting individual subviews include"]
#[doc = " [`.index_axis()`], [`.index_axis_mut()`], [`.index_axis_move()`], and"]
#[doc = " [`.index_axis_inplace()`]. You can also select a subview by using a single"]
#[doc = " index instead of a range when slicing. Some other methods, such as"]
#[doc = " [`.fold_axis()`], [`.axis_iter()`], [`.axis_iter_mut()`],"]
#[doc = " [`.outer_iter()`], and [`.outer_iter_mut()`] operate on all the subviews"]
#[doc = " along an axis."]
#[doc = ""]
#[doc = " A related method is [`.collapse_axis()`], which modifies the view in the"]
#[doc = " same way as [`.index_axis()`] except for removing the collapsed axis, since"]
#[doc = " it operates *in place*. The length of the axis becomes 1."]
#[doc = ""]
#[doc = " Methods for selecting an individual subview take two arguments: `axis` and"]
#[doc = " `index`."]
#[doc = ""]
#[doc = " [`.axis_iter()`]: Self::axis_iter"]
#[doc = " [`.axis_iter_mut()`]: Self::axis_iter_mut"]
#[doc = " [`.fold_axis()`]: Self::fold_axis"]
#[doc = " [`.index_axis()`]: Self::index_axis"]
#[doc = " [`.index_axis_inplace()`]: Self::index_axis_inplace"]
#[doc = " [`.index_axis_mut()`]: Self::index_axis_mut"]
#[doc = " [`.index_axis_move()`]: Self::index_axis_move"]
#[doc = " [`.collapse_axis()`]: Self::collapse_axis"]
#[doc = " [`.outer_iter()`]: Self::outer_iter"]
#[doc = " [`.outer_iter_mut()`]: Self::outer_iter_mut"]
#[doc = ""]
#[doc = " ```"]
#[doc = ""]
#[doc = " use ndarray::{arr3, aview1, aview2, s, Axis};"]
#[doc = ""]
#[doc = ""]
#[doc = " // 2 submatrices of 2 rows with 3 elements per row, means a shape of `[2, 2, 3]`."]
#[doc = ""]
#[doc = " let a = arr3(&[[[ 1,  2,  3],    // \\ axis 0, submatrix 0"]
#[doc = "                 [ 4,  5,  6]],   // /"]
#[doc = "                [[ 7,  8,  9],    // \\ axis 0, submatrix 1"]
#[doc = "                 [10, 11, 12]]]); // /"]
#[doc = "         //        \\"]
#[doc = "         //         axis 2, column 0"]
#[doc = ""]
#[doc = " assert_eq!(a.shape(), &[2, 2, 3]);"]
#[doc = ""]
#[doc = " // Let’s take a subview along the greatest dimension (axis 0),"]
#[doc = " // taking submatrix 0, then submatrix 1"]
#[doc = ""]
#[doc = " let sub_0 = a.index_axis(Axis(0), 0);"]
#[doc = " let sub_1 = a.index_axis(Axis(0), 1);"]
#[doc = ""]
#[doc = " assert_eq!(sub_0, aview2(&[[ 1,  2,  3],"]
#[doc = "                            [ 4,  5,  6]]));"]
#[doc = " assert_eq!(sub_1, aview2(&[[ 7,  8,  9],"]
#[doc = "                            [10, 11, 12]]));"]
#[doc = " assert_eq!(sub_0.shape(), &[2, 3]);"]
#[doc = ""]
#[doc = " // This is the subview picking only axis 2, column 0"]
#[doc = " let sub_col = a.index_axis(Axis(2), 0);"]
#[doc = ""]
#[doc = " assert_eq!(sub_col, aview2(&[[ 1,  4],"]
#[doc = "                              [ 7, 10]]));"]
#[doc = ""]
#[doc = " // You can take multiple subviews at once (and slice at the same time)"]
#[doc = " let double_sub = a.slice(s![1, .., 0]);"]
#[doc = " assert_eq!(double_sub, aview1(&[7, 10]));"]
#[doc = " ```"]
#[doc = ""]
#[doc = " ## Arithmetic Operations"]
#[doc = ""]
#[doc = " Arrays support all arithmetic operations the same way: they apply elementwise."]
#[doc = ""]
#[doc = " Since the trait implementations are hard to overview, here is a summary."]
#[doc = ""]
#[doc = " ### Binary Operators with Two Arrays"]
#[doc = ""]
#[doc = " Let `A` be an array or view of any kind. Let `B` be an array"]
#[doc = " with owned storage (either `Array` or `ArcArray`)."]
#[doc = " Let `C` be an array with mutable data (either `Array`, `ArcArray`"]
#[doc = " or `ArrayViewMut`)."]
#[doc = " The following combinations of operands"]
#[doc = " are supported for an arbitrary binary operator denoted by `@` (it can be"]
#[doc = " `+`, `-`, `*`, `/` and so on)."]
#[doc = ""]
#[doc = " - `&A @ &A` which produces a new `Array`"]
#[doc = " - `B @ A` which consumes `B`, updates it with the result, and returns it"]
#[doc = " - `B @ &A` which consumes `B`, updates it with the result, and returns it"]
#[doc = " - `C @= &A` which performs an arithmetic operation in place"]
#[doc = ""]
#[doc = " Note that the element type needs to implement the operator trait and the"]
#[doc = " `Clone` trait."]
#[doc = ""]
#[doc = " ```"]
#[doc = " use ndarray::{array, ArrayView1};"]
#[doc = ""]
#[doc = " let owned1 = array![1, 2];"]
#[doc = " let owned2 = array![3, 4];"]
#[doc = " let view1 = ArrayView1::from(&[5, 6]);"]
#[doc = " let view2 = ArrayView1::from(&[7, 8]);"]
#[doc = " let mut mutable = array![9, 10];"]
#[doc = ""]
#[doc = " let sum1 = &view1 + &view2;   // Allocates a new array. Note the explicit `&`."]
#[doc = " // let sum2 = view1 + &view2; // This doesn't work because `view1` is not an owned array."]
#[doc = " let sum3 = owned1 + view1;    // Consumes `owned1`, updates it, and returns it."]
#[doc = " let sum4 = owned2 + &view2;   // Consumes `owned2`, updates it, and returns it."]
#[doc = " mutable += &view2;            // Updates `mutable` in-place."]
#[doc = " ```"]
#[doc = ""]
#[doc = " ### Binary Operators with Array and Scalar"]
#[doc = ""]
#[doc = " The trait [`ScalarOperand`] marks types that can be used in arithmetic"]
#[doc = " with arrays directly. For a scalar `K` the following combinations of operands"]
#[doc = " are supported (scalar can be on either the left or right side, but"]
#[doc = " `ScalarOperand` docs has the detailed conditions)."]
#[doc = ""]
#[doc = " - `&A @ K` or `K @ &A` which produces a new `Array`"]
#[doc = " - `B @ K` or `K @ B` which consumes `B`, updates it with the result and returns it"]
#[doc = " - `C @= K` which performs an arithmetic operation in place"]
#[doc = ""]
#[doc = " ### Unary Operators"]
#[doc = ""]
#[doc = " Let `A` be an array or view of any kind. Let `B` be an array with owned"]
#[doc = " storage (either `Array` or `ArcArray`). The following operands are supported"]
#[doc = " for an arbitrary unary operator denoted by `@` (it can be `-` or `!`)."]
#[doc = ""]
#[doc = " - `@&A` which produces a new `Array`"]
#[doc = " - `@B` which consumes `B`, updates it with the result, and returns it"]
#[doc = ""]
#[doc = " ## Broadcasting"]
#[doc = ""]
#[doc = " Arrays support limited *broadcasting*, where arithmetic operations with"]
#[doc = " array operands of different sizes can be carried out by repeating the"]
#[doc = " elements of the smaller dimension array. See"]
#[doc = " [`.broadcast()`](Self::broadcast) for a more detailed"]
#[doc = " description."]
#[doc = ""]
#[doc = " ```"]
#[doc = " use ndarray::arr2;"]
#[doc = ""]
#[doc = " let a = arr2(&[[1., 1.],"]
#[doc = "                [1., 2.],"]
#[doc = "                [0., 3.],"]
#[doc = "                [0., 4.]]);"]
#[doc = ""]
#[doc = " let b = arr2(&[[0., 1.]]);"]
#[doc = ""]
#[doc = " let c = arr2(&[[1., 2.],"]
#[doc = "                [1., 3.],"]
#[doc = "                [0., 4.],"]
#[doc = "                [0., 5.]]);"]
#[doc = " // We can add because the shapes are compatible even if not equal."]
#[doc = " // The `b` array is shape 1 × 2 but acts like a 4 × 2 array."]
#[doc = " assert!("]
#[doc = "     c == a + b"]
#[doc = " );"]
#[doc = " ```"]
#[doc = ""]
#[doc = " ## Conversions"]
#[doc = ""]
#[doc = " ### Conversions Between Array Types"]
#[doc = ""]
#[doc = " This table is a summary of the conversions between arrays of different"]
#[doc = " ownership, dimensionality, and element type. All of the conversions in this"]
#[doc = " table preserve the shape of the array."]
#[doc = ""]
#[doc = " <table>"]
#[doc = " <tr>"]
#[doc = " <th rowspan=\"2\">Output</th>"]
#[doc = " <th colspan=\"5\">Input</th>"]
#[doc = " </tr>"]
#[doc = ""]
#[doc = " <tr>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `Array<A, D>`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `ArcArray<A, D>`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `CowArray<'a, A, D>`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `ArrayView<'a, A, D>`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `ArrayViewMut<'a, A, D>`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " </tr>"]
#[doc = ""]
#[doc = " <!--Conversions to `Array<A, D>`-->"]
#[doc = ""]
#[doc = " <tr>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `Array<A, D>`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " no-op"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.into_owned()`][.into_owned()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.into_owned()`][.into_owned()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.to_owned()`][.to_owned()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.to_owned()`][.to_owned()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " </tr>"]
#[doc = ""]
#[doc = " <!--Conversions to `ArcArray<A, D>`-->"]
#[doc = ""]
#[doc = " <tr>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `ArcArray<A, D>`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.into_shared()`][.into_shared()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " no-op"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.into_owned().into_shared()`][.into_shared()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.to_owned().into_shared()`][.into_shared()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.to_owned().into_shared()`][.into_shared()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " </tr>"]
#[doc = ""]
#[doc = " <!--Conversions to `CowArray<'a, A, D>`-->"]
#[doc = ""]
#[doc = " <tr>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `CowArray<'a, A, D>`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`CowArray::from(a)`](CowArray#impl-From<ArrayBase<OwnedRepr<A>%2C%20D>>)"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`CowArray::from(a.into_owned())`](CowArray#impl-From<ArrayBase<OwnedRepr<A>%2C%20D>>)"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " no-op"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`CowArray::from(a)`](CowArray#impl-From<ArrayBase<ViewRepr<%26%27a%20A>%2C%20D>>)"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`CowArray::from(a.view())`](CowArray#impl-From<ArrayBase<ViewRepr<%26%27a%20A>%2C%20D>>)"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " </tr>"]
#[doc = ""]
#[doc = " <!--Conversions to `ArrayView<'b, A, D>`-->"]
#[doc = ""]
#[doc = " <tr>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `ArrayView<'b, A, D>`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.view()`][.view()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.view()`][.view()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.view()`][.view()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.view()`][.view()] or [`a.reborrow()`][ArrayView::reborrow()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.view()`][.view()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " </tr>"]
#[doc = ""]
#[doc = " <!--Conversions to `ArrayViewMut<'b, A, D>`-->"]
#[doc = ""]
#[doc = " <tr>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `ArrayViewMut<'b, A, D>`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.view_mut()`][.view_mut()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.view_mut()`][.view_mut()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.view_mut()`][.view_mut()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " illegal"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " [`a.view_mut()`][.view_mut()] or [`a.reborrow()`][ArrayViewMut::reborrow()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " </tr>"]
#[doc = ""]
#[doc = " <!--Conversions to equivalent with dim `D2`-->"]
#[doc = ""]
#[doc = " <tr>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " equivalent with dim `D2` (e.g. converting from dynamic dim to const dim)"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td colspan=\"5\">"]
#[doc = ""]
#[doc = " [`a.into_dimensionality::<D2>()`][.into_dimensionality()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " </tr>"]
#[doc = ""]
#[doc = " <!--Conversions to equivalent with dim `IxDyn`-->"]
#[doc = ""]
#[doc = " <tr>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " equivalent with dim `IxDyn`"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td colspan=\"5\">"]
#[doc = ""]
#[doc = " [`a.into_dyn()`][.into_dyn()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " </tr>"]
#[doc = ""]
#[doc = " <!--Conversions to `Array<B, D>`-->"]
#[doc = ""]
#[doc = " <tr>"]
#[doc = " <td>"]
#[doc = ""]
#[doc = " `Array<B, D>` (new element type)"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " <td colspan=\"5\">"]
#[doc = ""]
#[doc = " [`a.map(|x| x.do_your_conversion())`][.map()]"]
#[doc = ""]
#[doc = " </td>"]
#[doc = " </tr>"]
#[doc = " </table>"]
#[doc = ""]
#[doc = " ### Conversions Between Arrays and `Vec`s/Slices/Scalars"]
#[doc = ""]
#[doc = " This is a table of the safe conversions between arrays and"]
#[doc = " `Vec`s/slices/scalars. Note that some of the return values are actually"]
#[doc = " `Result`/`Option` wrappers around the indicated output types."]
#[doc = ""]
#[doc = " Input | Output | Methods"]
#[doc = " ------|--------|--------"]
#[doc = " `Vec<A>` | `ArrayBase<S: DataOwned, Ix1>` | [`::from_vec()`](Self::from_vec)"]
#[doc = " `Vec<A>` | `ArrayBase<S: DataOwned, D>` | [`::from_shape_vec()`](Self::from_shape_vec)"]
#[doc = " `&[A]` | `ArrayView1<A>` | [`::from()`](ArrayView#method.from)"]
#[doc = " `&[A]` | `ArrayView<A, D>` | [`::from_shape()`](ArrayView#method.from_shape)"]
#[doc = " `&mut [A]` | `ArrayViewMut1<A>` | [`::from()`](ArrayViewMut#method.from)"]
#[doc = " `&mut [A]` | `ArrayViewMut<A, D>` | [`::from_shape()`](ArrayViewMut#method.from_shape)"]
#[doc = " `&ArrayBase<S, Ix1>` | `Vec<A>` | [`.to_vec()`](Self::to_vec)"]
#[doc = " `Array<A, D>` | `Vec<A>` | [`.into_raw_vec()`](Array#method.into_raw_vec)<sup>[1](#into_raw_vec)</sup>"]
#[doc = " `&ArrayBase<S, D>` | `&[A]` | [`.as_slice()`](Self::as_slice)<sup>[2](#req_contig_std)</sup>, [`.as_slice_memory_order()`](Self::as_slice_memory_order)<sup>[3](#req_contig)</sup>"]
#[doc = " `&mut ArrayBase<S: DataMut, D>` | `&mut [A]` | [`.as_slice_mut()`](Self::as_slice_mut)<sup>[2](#req_contig_std)</sup>, [`.as_slice_memory_order_mut()`](Self::as_slice_memory_order_mut)<sup>[3](#req_contig)</sup>"]
#[doc = " `ArrayView<A, D>` | `&[A]` | [`.to_slice()`](ArrayView#method.to_slice)<sup>[2](#req_contig_std)</sup>"]
#[doc = " `ArrayViewMut<A, D>` | `&mut [A]` | [`.into_slice()`](ArrayViewMut#method.into_slice)<sup>[2](#req_contig_std)</sup>"]
#[doc = " `Array0<A>` | `A` | [`.into_scalar()`](Array#method.into_scalar)"]
#[doc = ""]
#[doc = " <sup><a name=\"into_raw_vec\">1</a></sup>Returns the data in memory order."]
#[doc = ""]
#[doc = " <sup><a name=\"req_contig_std\">2</a></sup>Works only if the array is"]
#[doc = " contiguous and in standard order."]
#[doc = ""]
#[doc = " <sup><a name=\"req_contig\">3</a></sup>Works only if the array is contiguous."]
#[doc = ""]
#[doc = " The table above does not include all the constructors; it only shows"]
#[doc = " conversions to/from `Vec`s/slices. See"]
#[doc = " [below](#constructor-methods-for-owned-arrays) for more constructors."]
#[doc = ""]
#[doc = " [ArrayView::reborrow()]: ArrayView#method.reborrow"]
#[doc = " [ArrayViewMut::reborrow()]: ArrayViewMut#method.reborrow"]
#[doc = " [.into_dimensionality()]: Self::into_dimensionality"]
#[doc = " [.into_dyn()]: Self::into_dyn"]
#[doc = " [.into_owned()]: Self::into_owned"]
#[doc = " [.into_shared()]: Self::into_shared"]
#[doc = " [.to_owned()]: Self::to_owned"]
#[doc = " [.map()]: Self::map"]
#[doc = " [.view()]: Self::view"]
#[doc = " [.view_mut()]: Self::view_mut"]
#[doc = ""]
#[doc = " ### Conversions from Nested `Vec`s/`Array`s"]
#[doc = ""]
#[doc = " It's generally a good idea to avoid nested `Vec`/`Array` types, such as"]
#[doc = " `Vec<Vec<A>>` or `Vec<Array2<A>>` because:"]
#[doc = ""]
#[doc = " * they require extra heap allocations compared to a single `Array`,"]
#[doc = ""]
#[doc = " * they can scatter data all over memory (because of multiple allocations),"]
#[doc = ""]
#[doc = " * they cause unnecessary indirection (traversing multiple pointers to reach"]
#[doc = "   the data),"]
#[doc = ""]
#[doc = " * they don't enforce consistent shape within the nested"]
#[doc = "   `Vec`s/`ArrayBase`s, and"]
#[doc = ""]
#[doc = " * they are generally more difficult to work with."]
#[doc = ""]
#[doc = " The most common case where users might consider using nested"]
#[doc = " `Vec`s/`Array`s is when creating an array by appending rows/subviews in a"]
#[doc = " loop, where the rows/subviews are computed within the loop. However, there"]
#[doc = " are better ways than using nested `Vec`s/`Array`s."]
#[doc = ""]
#[doc = " If you know ahead-of-time the shape of the final array, the cleanest"]
#[doc = " solution is to allocate the final array before the loop, and then assign"]
#[doc = " the data to it within the loop, like this:"]
#[doc = ""]
#[doc = " ```rust"]
#[doc = " use ndarray::{array, Array2, Axis};"]
#[doc = ""]
#[doc = " let mut arr = Array2::zeros((2, 3));"]
#[doc = " for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {"]
#[doc = "     // Perform calculations and assign to `row`; this is a trivial example:"]
#[doc = "     row.fill(i);"]
#[doc = " }"]
#[doc = " assert_eq!(arr, array![[0, 0, 0], [1, 1, 1]]);"]
#[doc = " ```"]
#[doc = ""]
#[doc = " If you don't know ahead-of-time the shape of the final array, then the"]
#[doc = " cleanest solution is generally to append the data to a flat `Vec`, and then"]
#[doc = " convert it to an `Array` at the end with"]
#[doc = " [`::from_shape_vec()`](Self::from_shape_vec). You just have to be careful"]
#[doc = " that the layout of the data (the order of the elements in the flat `Vec`)"]
#[doc = " is correct."]
#[doc = ""]
#[doc = " ```rust"]
#[doc = " use ndarray::{array, Array2};"]
#[doc = ""]
#[doc = " let ncols = 3;"]
#[doc = " let mut data = Vec::new();"]
#[doc = " let mut nrows = 0;"]
#[doc = " for i in 0..2 {"]
#[doc = "     // Compute `row` and append it to `data`; this is a trivial example:"]
#[doc = "     let row = vec![i; ncols];"]
#[doc = "     data.extend_from_slice(&row);"]
#[doc = "     nrows += 1;"]
#[doc = " }"]
#[doc = " let arr = Array2::from_shape_vec((nrows, ncols), data)?;"]
#[doc = " assert_eq!(arr, array![[0, 0, 0], [1, 1, 1]]);"]
#[doc = " # Ok::<(), ndarray::ShapeError>(())"]
#[doc = " ```"]
#[doc = ""]
#[doc = " If neither of these options works for you, and you really need to convert"]
#[doc = " nested `Vec`/`Array` instances to an `Array`, the cleanest solution is"]
#[doc = " generally to use [`Iterator::flatten()`]"]
#[doc = " to get a flat `Vec`, and then convert the `Vec` to an `Array` with"]
#[doc = " [`::from_shape_vec()`](Self::from_shape_vec), like this:"]
#[doc = ""]
#[doc = " ```rust"]
#[doc = " use ndarray::{array, Array2, Array3};"]
#[doc = ""]
#[doc = " let nested: Vec<Array2<i32>> = vec!["]
#[doc = "     array![[1, 2, 3], [4, 5, 6]],"]
#[doc = "     array![[7, 8, 9], [10, 11, 12]],"]
#[doc = " ];"]
#[doc = " let inner_shape = nested[0].dim();"]
#[doc = " let shape = (nested.len(), inner_shape.0, inner_shape.1);"]
#[doc = " let flat: Vec<i32> = nested.iter().flatten().cloned().collect();"]
#[doc = " let arr = Array3::from_shape_vec(shape, flat)?;"]
#[doc = " assert_eq!(arr, array!["]
#[doc = "     [[1, 2, 3], [4, 5, 6]],"]
#[doc = "     [[7, 8, 9], [10, 11, 12]],"]
#[doc = " ]);"]
#[doc = " # Ok::<(), ndarray::ShapeError>(())"]
#[doc = " ```"]
#[doc = ""]
#[doc = " Note that this implementation assumes that the nested `Vec`s are all the"]
#[doc = " same shape and that the `Vec` is non-empty. Depending on your application,"]
#[doc = " it may be a good idea to add checks for these assumptions and possibly"]
#[doc = " choose a different way to handle the empty case."]
#[doc = ""]
pub struct ArrayBase<S, D>
where
    S: RawData,
{
    #[doc = " Data buffer / ownership information. (If owned, contains the data"]
    #[doc = " buffer; if borrowed, contains the lifetime and mutability.)"]
    data: S,
    #[doc = " A non-null pointer into the buffer held by `data`; may point anywhere"]
    #[doc = " in its range. If `S: Data`, this pointer must be aligned."]
    ptr: std::ptr::NonNull<S::Elem>,
    #[doc = " The lengths of the axes."]
    dim: D,
    #[doc = " The element count stride per axis. To be parsed as `isize`."]
    strides: D,
}
#[doc = " An array where the data has shared ownership and is copy on write."]
#[doc = ""]
#[doc = " The `ArcArray<A, D>` is parameterized by `A` for the element type and `D` for"]
#[doc = " the dimensionality."]
#[doc = ""]
#[doc = " It can act as both an owner as the data as well as a shared reference (view"]
#[doc = " like)."]
#[doc = " Calling a method for mutating elements on `ArcArray`, for example"]
#[doc = " [`view_mut()`](ArrayBase::view_mut) or"]
#[doc = " [`get_mut()`](ArrayBase::get_mut), will break sharing and"]
#[doc = " require a clone of the data (if it is not uniquely held)."]
#[doc = ""]
#[doc = " `ArcArray` uses atomic reference counting like `Arc`, so it is `Send` and"]
#[doc = " `Sync` (when allowed by the element type of the array too)."]
#[doc = ""]
#[doc = " **[`ArrayBase`]** is used to implement both the owned"]
#[doc = " arrays and the views; see its docs for an overview of all array features."]
#[doc = ""]
#[doc = " See also:"]
#[doc = ""]
#[doc = " + [Constructor Methods for Owned Arrays](ArrayBase#constructor-methods-for-owned-arrays)"]
#[doc = " + [Methods For All Array Types](ArrayBase#methods-for-all-array-types)"]
pub type ArcArray<A, D> = ArrayBase<OwnedArcRepr<A>, D>;
#[doc = " An array that owns its data uniquely."]
#[doc = ""]
#[doc = " `Array` is the main n-dimensional array type, and it owns all its array"]
#[doc = " elements."]
#[doc = ""]
#[doc = " The `Array<A, D>` is parameterized by `A` for the element type and `D` for"]
#[doc = " the dimensionality."]
#[doc = ""]
#[doc = " **[`ArrayBase`]** is used to implement both the owned"]
#[doc = " arrays and the views; see its docs for an overview of all array features."]
#[doc = ""]
#[doc = " See also:"]
#[doc = ""]
#[doc = " + [Constructor Methods for Owned Arrays](ArrayBase#constructor-methods-for-owned-arrays)"]
#[doc = " + [Methods For All Array Types](ArrayBase#methods-for-all-array-types)"]
#[doc = " + Dimensionality-specific type alises"]
#[doc = " [`Array1`],"]
#[doc = " [`Array2`],"]
#[doc = " [`Array3`], ...,"]
#[doc = " [`ArrayD`],"]
#[doc = " and so on."]
pub type Array<A, D> = ArrayBase<OwnedRepr<A>, D>;
#[doc = " An array with copy-on-write behavior."]
#[doc = ""]
#[doc = " An `CowArray` represents either a uniquely owned array or a view of an"]
#[doc = " array. The `'a` corresponds to the lifetime of the view variant."]
#[doc = ""]
#[doc = " This type is analogous to [`std::borrow::Cow`]."]
#[doc = " If a `CowArray` instance is the immutable view variant, then calling a"]
#[doc = " method for mutating elements in the array will cause it to be converted"]
#[doc = " into the owned variant (by cloning all the elements) before the"]
#[doc = " modification is performed."]
#[doc = ""]
#[doc = " Array views have all the methods of an array (see [`ArrayBase`])."]
#[doc = ""]
#[doc = " See also [`ArcArray`], which also provides"]
#[doc = " copy-on-write behavior but has a reference-counted pointer to the data"]
#[doc = " instead of either a view or a uniquely owned copy."]
pub type CowArray<'a, A, D> = ArrayBase<CowRepr<'a, A>, D>;
#[doc = " A read-only array view."]
#[doc = ""]
#[doc = " An array view represents an array or a part of it, created from"]
#[doc = " an iterator, subview or slice of an array."]
#[doc = ""]
#[doc = " The `ArrayView<'a, A, D>` is parameterized by `'a` for the scope of the"]
#[doc = " borrow, `A` for the element type and `D` for the dimensionality."]
#[doc = ""]
#[doc = " Array views have all the methods of an array (see [`ArrayBase`])."]
#[doc = ""]
#[doc = " See also [`ArrayViewMut`]."]
pub type ArrayView<'a, A, D> = ArrayBase<ViewRepr<&'a A>, D>;
#[doc = " A read-write array view."]
#[doc = ""]
#[doc = " An array view represents an array or a part of it, created from"]
#[doc = " an iterator, subview or slice of an array."]
#[doc = ""]
#[doc = " The `ArrayViewMut<'a, A, D>` is parameterized by `'a` for the scope of the"]
#[doc = " borrow, `A` for the element type and `D` for the dimensionality."]
#[doc = ""]
#[doc = " Array views have all the methods of an array (see [`ArrayBase`])."]
#[doc = ""]
#[doc = " See also [`ArrayView`]."]
pub type ArrayViewMut<'a, A, D> = ArrayBase<ViewRepr<&'a mut A>, D>;
#[doc = " A read-only array view without a lifetime."]
#[doc = ""]
#[doc = " This is similar to [`ArrayView`] but does not carry any lifetime or"]
#[doc = " ownership information, and its data cannot be read without an unsafe"]
#[doc = " conversion into an [`ArrayView`]. The relationship between `RawArrayView`"]
#[doc = " and [`ArrayView`] is somewhat analogous to the relationship between `*const"]
#[doc = " T` and `&T`, but `RawArrayView` has additional requirements that `*const T`"]
#[doc = " does not, such as non-nullness."]
#[doc = ""]
#[doc = " The `RawArrayView<A, D>` is parameterized by `A` for the element type and"]
#[doc = " `D` for the dimensionality."]
#[doc = ""]
#[doc = " Raw array views have all the methods of an array (see"]
#[doc = " [`ArrayBase`])."]
#[doc = ""]
#[doc = " See also [`RawArrayViewMut`]."]
#[doc = ""]
#[doc = " # Warning"]
#[doc = ""]
#[doc = " You can't use this type with an arbitrary raw pointer; see"]
#[doc = " [`from_shape_ptr`](#method.from_shape_ptr) for details."]
pub type RawArrayView<A, D> = ArrayBase<RawViewRepr<*const A>, D>;
#[doc = " A mutable array view without a lifetime."]
#[doc = ""]
#[doc = " This is similar to [`ArrayViewMut`] but does not carry any lifetime or"]
#[doc = " ownership information, and its data cannot be read/written without an"]
#[doc = " unsafe conversion into an [`ArrayViewMut`]. The relationship between"]
#[doc = " `RawArrayViewMut` and [`ArrayViewMut`] is somewhat analogous to the"]
#[doc = " relationship between `*mut T` and `&mut T`, but `RawArrayViewMut` has"]
#[doc = " additional requirements that `*mut T` does not, such as non-nullness."]
#[doc = ""]
#[doc = " The `RawArrayViewMut<A, D>` is parameterized by `A` for the element type"]
#[doc = " and `D` for the dimensionality."]
#[doc = ""]
#[doc = " Raw array views have all the methods of an array (see"]
#[doc = " [`ArrayBase`])."]
#[doc = ""]
#[doc = " See also [`RawArrayView`]."]
#[doc = ""]
#[doc = " # Warning"]
#[doc = ""]
#[doc = " You can't use this type with an arbitrary raw pointer; see"]
#[doc = " [`from_shape_ptr`](#method.from_shape_ptr) for details."]
pub type RawArrayViewMut<A, D> = ArrayBase<RawViewRepr<*mut A>, D>;
pub use data_repr::OwnedRepr;
#[doc = " ArcArray's representation."]
#[doc = ""]
#[doc = " *Don’t use this type directly—use the type alias"]
#[doc = " [`ArcArray`] for the array type!*"]
#[derive(Debug)]
pub struct OwnedArcRepr<A>(Arc<OwnedRepr<A>>);
impl<A> Clone for OwnedArcRepr<A> {
    fn clone(&self) -> Self {
        OwnedArcRepr(self.0.clone())
    }
}
#[doc = " Array pointer’s representation."]
#[doc = ""]
#[doc = " *Don’t use this type directly—use the type aliases"]
#[doc = " [`RawArrayView`] / [`RawArrayViewMut`] for the array type!*"]
#[derive(Copy, Clone)]
pub struct RawViewRepr<A> {
    ptr: PhantomData<A>,
}
impl<A> RawViewRepr<A> {
    #[inline(always)]
    fn new() -> Self {
        RawViewRepr { ptr: PhantomData }
    }
}
#[doc = " Array view’s representation."]
#[doc = ""]
#[doc = " *Don’t use this type directly—use the type aliases"]
#[doc = " [`ArrayView`] / [`ArrayViewMut`] for the array type!*"]
#[derive(Copy, Clone)]
pub struct ViewRepr<A> {
    life: PhantomData<A>,
}
impl<A> ViewRepr<A> {
    #[inline(always)]
    fn new() -> Self {
        ViewRepr { life: PhantomData }
    }
}
#[doc = " CowArray's representation."]
#[doc = ""]
#[doc = " *Don't use this type directly—use the type alias"]
#[doc = " [`CowArray`] for the array type!*"]
pub enum CowRepr<'a, A> {
    #[doc = " Borrowed data."]
    View(ViewRepr<&'a A>),
    #[doc = " Owned data."]
    Owned(OwnedRepr<A>),
}
impl<'a, A> CowRepr<'a, A> {
    #[doc = " Returns `true` iff the data is the `View` variant."]
    pub fn is_view(&self) -> bool {
        match self {
            CowRepr::View(_) => true,
            CowRepr::Owned(_) => false,
        }
    }
    #[doc = " Returns `true` iff the data is the `Owned` variant."]
    pub fn is_owned(&self) -> bool {
        match self {
            CowRepr::View(_) => false,
            CowRepr::Owned(_) => true,
        }
    }
}
mod impl_clone {
    use crate::imp_prelude::*;
    use crate::RawDataClone;
    impl<S: RawDataClone, D: Clone> Clone for ArrayBase<S, D> {
        fn clone(&self) -> ArrayBase<S, D> {
            unsafe {
                let (data, ptr) = self.data.clone_with_ptr(self.ptr);
                ArrayBase {
                    data,
                    ptr,
                    dim: self.dim.clone(),
                    strides: self.strides.clone(),
                }
            }
        }
    }
}
mod impl_internal_constructors {
    use crate::imp_prelude::*;
    use std::ptr::NonNull;
    impl<A, S> ArrayBase<S, Ix1>
    where
        S: RawData<Elem = A>,
    {
        #[doc = " Create an (initially) empty one-dimensional array from the given data and array head"]
        #[doc = " pointer"]
        #[doc = ""]
        #[doc = " ## Safety"]
        #[doc = ""]
        #[doc = " The caller must ensure that the data storage and pointer is valid."]
        #[doc = " "]
        #[doc = " See ArrayView::from_shape_ptr for general pointer validity documentation."]
        pub(crate) unsafe fn from_data_ptr(data: S, ptr: NonNull<A>) -> Self {
            let array = ArrayBase {
                data,
                ptr,
                dim: Ix1(0),
                strides: Ix1(1),
            };
            debug_assert!(array.pointer_is_inbounds());
            array
        }
    }
    impl<A, S, D> ArrayBase<S, D>
    where
        S: RawData<Elem = A>,
        D: Dimension,
    {
        #[doc = " Set strides and dimension of the array to the new values"]
        #[doc = ""]
        #[doc = " The argument order with strides before dimensions is used because strides are often"]
        #[doc = " computed as derived from the dimension."]
        #[doc = ""]
        #[doc = " ## Safety"]
        #[doc = ""]
        #[doc = " The caller needs to ensure that the new strides and dimensions are correct"]
        #[doc = " for the array data."]
        pub(crate) unsafe fn with_strides_dim<E>(self, strides: E, dim: E) -> ArrayBase<S, E>
        where
            E: Dimension,
        {
            debug_assert_eq!(strides.ndim(), dim.ndim());
            ArrayBase {
                data: self.data,
                ptr: self.ptr,
                dim,
                strides,
            }
        }
    }
}
mod impl_constructors {
    #![doc = " Constructor methods for ndarray"]
    #![doc = ""]
    #![doc = ""]
    #![allow(clippy::match_wild_err_arm)]
    use crate::dimension;
    use crate::dimension::offset_from_low_addr_ptr_to_logical_ptr;
    use crate::error::{self, ShapeError};
    use crate::extension::nonnull::nonnull_from_vec_data;
    use crate::imp_prelude::*;
    use crate::indexes;
    use crate::indices;
    #[cfg(feature = "std")]
    use crate::iterators::to_vec;
    use crate::iterators::to_vec_mapped;
    use crate::iterators::TrustedIterator;
    use crate::StrideShape;
    #[cfg(feature = "std")]
    use crate::{geomspace, linspace, logspace};
    use alloc::vec;
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use num_traits::Float;
    use num_traits::{One, Zero};
    use rawpointer::PointerExt;
    use std::mem;
    use std::mem::MaybeUninit;
    #[doc = " # Constructor Methods for Owned Arrays"]
    #[doc = ""]
    #[doc = " Note that the constructor methods apply to `Array` and `ArcArray`,"]
    #[doc = " the two array types that have owned storage."]
    #[doc = ""]
    #[doc = " ## Constructor methods for one-dimensional arrays."]
    impl<S, A> ArrayBase<S, Ix1>
    where
        S: DataOwned<Elem = A>,
    {
        #[doc = " Create a one-dimensional array from a vector (no copying needed)."]
        #[doc = ""]
        #[doc = " **Panics** if the length is greater than `isize::MAX`."]
        #[doc = ""]
        #[doc = " ```rust"]
        #[doc = " use ndarray::Array;"]
        #[doc = ""]
        #[doc = " let array = Array::from_vec(vec![1., 2., 3., 4.]);"]
        #[doc = " ```"]
        pub fn from_vec(v: Vec<A>) -> Self {
            if mem::size_of::<A>() == 0 {
                assert!(
                    v.len() <= isize::MAX as usize,
                    "Length must fit in `isize`.",
                );
            }
            unsafe { Self::from_shape_vec_unchecked(v.len() as Ix, v) }
        }
        #[doc = " Create a one-dimensional array from an iterator or iterable."]
        #[doc = ""]
        #[doc = " **Panics** if the length is greater than `isize::MAX`."]
        #[doc = ""]
        #[doc = " ```rust"]
        #[doc = " use ndarray::Array;"]
        #[doc = ""]
        #[doc = " let array = Array::from_iter(0..10);"]
        #[doc = " ```"]
        #[allow(clippy::should_implement_trait)]
        pub fn from_iter<I: IntoIterator<Item = A>>(iterable: I) -> Self {
            Self::from_vec(iterable.into_iter().collect())
        }
    }
    #[cfg(not(debug_assertions))]
    #[allow(clippy::match_wild_err_arm)]
    macro_rules ! size_of_shape_checked_unwrap { ($ dim : expr) => { match dimension :: size_of_shape_checked ($ dim) { Ok (sz) => sz , Err (_) => { panic ! ("ndarray: Shape too large, product of non-zero axis lengths overflows isize") } } } ; }
    #[cfg(debug_assertions)]
    macro_rules! size_of_shape_checked_unwrap {
        ($ dim : expr) => {
            match dimension::size_of_shape_checked($dim) {
                Ok(sz) => sz,
                Err(_) => panic!(
                    "ndarray: Shape too large, product of non-zero axis lengths \
                 overflows isize in shape {:?}",
                    $dim
                ),
            }
        };
    }
    #[doc = " ## Constructor methods for n-dimensional arrays."]
    #[doc = ""]
    #[doc = " The `shape` argument can be an integer or a tuple of integers to specify"]
    #[doc = " a static size. For example `10` makes a length 10 one-dimensional array"]
    #[doc = " (dimension type `Ix1`) and `(5, 6)` a 5 × 6 array (dimension type `Ix2`)."]
    #[doc = ""]
    #[doc = " With the trait `ShapeBuilder` in scope, there is the method `.f()` to select"]
    #[doc = " column major (“f” order) memory layout instead of the default row major."]
    #[doc = " For example `Array::zeros((5, 6).f())` makes a column major 5 × 6 array."]
    #[doc = ""]
    #[doc = " Use [`type@IxDyn`] for the shape to create an array with dynamic"]
    #[doc = " number of axes."]
    #[doc = ""]
    #[doc = " Finally, the few constructors that take a completely general"]
    #[doc = " `Into<StrideShape>` argument *optionally* support custom strides, for"]
    #[doc = " example a shape given like `(10, 2, 2).strides((1, 10, 20))` is valid."]
    impl<S, A, D> ArrayBase<S, D>
    where
        S: DataOwned<Elem = A>,
        D: Dimension,
    {
        #[doc = " Create an array with copies of `elem`, shape `shape`."]
        #[doc = ""]
        #[doc = " **Panics** if the product of non-zero axis lengths overflows `isize`."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::{Array, arr3, ShapeBuilder};"]
        #[doc = ""]
        #[doc = " let a = Array::from_elem((2, 2, 2), 1.);"]
        #[doc = ""]
        #[doc = " assert!("]
        #[doc = "     a == arr3(&[[[1., 1.],"]
        #[doc = "                  [1., 1.]],"]
        #[doc = "                 [[1., 1.],"]
        #[doc = "                  [1., 1.]]])"]
        #[doc = " );"]
        #[doc = " assert!(a.strides() == &[4, 2, 1]);"]
        #[doc = ""]
        #[doc = " let b = Array::from_elem((2, 2, 2).f(), 1.);"]
        #[doc = " assert!(b.strides() == &[1, 2, 4]);"]
        #[doc = " ```"]
        pub fn from_elem<Sh>(shape: Sh, elem: A) -> Self
        where
            A: Clone,
            Sh: ShapeBuilder<Dim = D>,
        {
            let shape = shape.into_shape();
            let size = size_of_shape_checked_unwrap!(&shape.dim);
            let v = vec![elem; size];
            unsafe { Self::from_shape_vec_unchecked(shape, v) }
        }
        #[doc = " Create an array with values created by the function `f`."]
        #[doc = ""]
        #[doc = " `f` is called with no argument, and it should return the element to"]
        #[doc = " create. If the precise index of the element to create is needed,"]
        #[doc = " use [`from_shape_fn`](ArrayBase::from_shape_fn) instead."]
        #[doc = ""]
        #[doc = " This constructor can be useful if the element order is not important,"]
        #[doc = " for example if they are identical or random."]
        #[doc = ""]
        #[doc = " **Panics** if the product of non-zero axis lengths overflows `isize`."]
        pub fn from_shape_simple_fn<Sh, F>(shape: Sh, mut f: F) -> Self
        where
            Sh: ShapeBuilder<Dim = D>,
            F: FnMut() -> A,
        {
            let shape = shape.into_shape();
            let len = size_of_shape_checked_unwrap!(&shape.dim);
            let v = to_vec_mapped(0..len, move |_| f());
            unsafe { Self::from_shape_vec_unchecked(shape, v) }
        }
        #[doc = " Create an array with values created by the function `f`."]
        #[doc = ""]
        #[doc = " `f` is called with the index of the element to create; the elements are"]
        #[doc = " visited in arbitrary order."]
        #[doc = ""]
        #[doc = " **Panics** if the product of non-zero axis lengths overflows `isize`."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::{Array, arr2};"]
        #[doc = ""]
        #[doc = " // Create a table of i ×\u{a0}j (with i and j from 1 to 3)"]
        #[doc = " let ij_table = Array::from_shape_fn((3, 3), |(i, j)| (1 + i) * (1 + j));"]
        #[doc = ""]
        #[doc = " assert_eq!("]
        #[doc = "     ij_table,"]
        #[doc = "     arr2(&[[1, 2, 3],"]
        #[doc = "            [2, 4, 6],"]
        #[doc = "            [3, 6, 9]])"]
        #[doc = " );"]
        #[doc = " ```"]
        pub fn from_shape_fn<Sh, F>(shape: Sh, f: F) -> Self
        where
            Sh: ShapeBuilder<Dim = D>,
            F: FnMut(D::Pattern) -> A,
        {
            let shape = shape.into_shape();
            let _ = size_of_shape_checked_unwrap!(&shape.dim);
            if shape.is_c() {
                let v = to_vec_mapped(indices(shape.dim.clone()).into_iter(), f);
                unsafe { Self::from_shape_vec_unchecked(shape, v) }
            } else {
                let dim = shape.dim.clone();
                let v = to_vec_mapped(indexes::indices_iter_f(dim), f);
                unsafe { Self::from_shape_vec_unchecked(shape, v) }
            }
        }
        #[doc = " Creates an array from a vector and interpret it according to the"]
        #[doc = " provided shape and strides. (No cloning of elements needed.)"]
        #[doc = ""]
        #[doc = " # Safety"]
        #[doc = ""]
        #[doc = " The caller must ensure that the following conditions are met:"]
        #[doc = ""]
        #[doc = " 1. The ndim of `dim` and `strides` must be the same."]
        #[doc = ""]
        #[doc = " 2. The product of non-zero axis lengths must not exceed `isize::MAX`."]
        #[doc = ""]
        #[doc = " 3. For axes with length > 1, the pointer cannot move outside the"]
        #[doc = "    slice."]
        #[doc = ""]
        #[doc = " 4. If the array will be empty (any axes are zero-length), the"]
        #[doc = "    difference between the least address and greatest address accessible"]
        #[doc = "    by moving along all axes must be ≤ `v.len()`."]
        #[doc = ""]
        #[doc = "    If the array will not be empty, the difference between the least"]
        #[doc = "    address and greatest address accessible by moving along all axes"]
        #[doc = "    must be < `v.len()`."]
        #[doc = ""]
        #[doc = " 5. The strides must not allow any element to be referenced by two different"]
        #[doc = "    indices."]
        pub unsafe fn from_shape_vec_unchecked<Sh>(shape: Sh, v: Vec<A>) -> Self
        where
            Sh: Into<StrideShape<D>>,
        {
            let shape = shape.into();
            let dim = shape.dim;
            let strides = shape.strides.strides_for_dim(&dim);
            Self::from_vec_dim_stride_unchecked(dim, strides, v)
        }
        unsafe fn from_vec_dim_stride_unchecked(dim: D, strides: D, mut v: Vec<A>) -> Self {
            debug_assert!(dimension::can_index_slice(&v, &dim, &strides).is_ok());
            let ptr = nonnull_from_vec_data(&mut v)
                .add(offset_from_low_addr_ptr_to_logical_ptr(&dim, &strides));
            ArrayBase::from_data_ptr(DataOwned::new(v), ptr).with_strides_dim(strides, dim)
        }
        #[doc = " Creates an array from an iterator, mapped by `map` and interpret it according to the"]
        #[doc = " provided shape and strides."]
        #[doc = ""]
        #[doc = " # Safety"]
        #[doc = ""]
        #[doc = " See from_shape_vec_unchecked"]
        pub(crate) unsafe fn from_shape_trusted_iter_unchecked<Sh, I, F>(
            shape: Sh,
            iter: I,
            map: F,
        ) -> Self
        where
            Sh: Into<StrideShape<D>>,
            I: TrustedIterator + ExactSizeIterator,
            F: FnMut(I::Item) -> A,
        {
            let shape = shape.into();
            let dim = shape.dim;
            let strides = shape.strides.strides_for_dim(&dim);
            let v = to_vec_mapped(iter, map);
            Self::from_vec_dim_stride_unchecked(dim, strides, v)
        }
        #[doc = " Create an array with uninitialized elements, shape `shape`."]
        #[doc = ""]
        #[doc = " The uninitialized elements of type `A` are represented by the type `MaybeUninit<A>`,"]
        #[doc = " an easier way to handle uninit values correctly."]
        #[doc = ""]
        #[doc = " Only *when* the array is completely initialized with valid elements, can it be"]
        #[doc = " converted to an array of `A` elements using [`.assume_init()`]."]
        #[doc = ""]
        #[doc = " **Panics** if the number of elements in `shape` would overflow isize."]
        #[doc = ""]
        #[doc = " ### Safety"]
        #[doc = ""]
        #[doc = " The whole of the array must be initialized before it is converted"]
        #[doc = " using [`.assume_init()`] or otherwise traversed/read with the element type `A`."]
        #[doc = ""]
        #[doc = " ### Examples"]
        #[doc = ""]
        #[doc = " It is possible to assign individual values through `*elt = MaybeUninit::new(value)`"]
        #[doc = " and so on."]
        #[doc = ""]
        #[doc = " [`.assume_init()`]: ArrayBase::assume_init"]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::{s, Array2};"]
        #[doc = ""]
        #[doc = " // Example Task: Let's create a column shifted copy of the input"]
        #[doc = ""]
        #[doc = " fn shift_by_two(a: &Array2<f32>) -> Array2<f32> {"]
        #[doc = "     // create an uninitialized array"]
        #[doc = "     let mut b = Array2::uninit(a.dim());"]
        #[doc = ""]
        #[doc = "     // two first columns in b are two last in a"]
        #[doc = "     // rest of columns in b are the initial columns in a"]
        #[doc = ""]
        #[doc = "     a.slice(s![.., -2..]).assign_to(b.slice_mut(s![.., ..2]));"]
        #[doc = "     a.slice(s![.., 2..]).assign_to(b.slice_mut(s![.., ..-2]));"]
        #[doc = ""]
        #[doc = "     // Now we can promise that `b` is safe to use with all operations"]
        #[doc = "     unsafe {"]
        #[doc = "         b.assume_init()"]
        #[doc = "     }"]
        #[doc = " }"]
        #[doc = " "]
        #[doc = " # let _ = shift_by_two;"]
        #[doc = " ```"]
        pub fn uninit<Sh>(shape: Sh) -> ArrayBase<S::MaybeUninit, D>
        where
            Sh: ShapeBuilder<Dim = D>,
        {
            unsafe {
                let shape = shape.into_shape();
                let size = size_of_shape_checked_unwrap!(&shape.dim);
                let mut v = Vec::with_capacity(size);
                v.set_len(size);
                ArrayBase::from_shape_vec_unchecked(shape, v)
            }
        }
        #[doc = " Create an array with uninitialized elements, shape `shape`."]
        #[doc = ""]
        #[doc = " The uninitialized elements of type `A` are represented by the type `MaybeUninit<A>`,"]
        #[doc = " an easier way to handle uninit values correctly."]
        #[doc = ""]
        #[doc = " The `builder` closure gets unshared access to the array through a view and can use it to"]
        #[doc = " modify the array before it is returned. This allows initializing the array for any owned"]
        #[doc = " array type (avoiding clone requirements for copy-on-write, because the array is unshared"]
        #[doc = " when initially created)."]
        #[doc = ""]
        #[doc = " Only *when* the array is completely initialized with valid elements, can it be"]
        #[doc = " converted to an array of `A` elements using [`.assume_init()`]."]
        #[doc = ""]
        #[doc = " **Panics** if the number of elements in `shape` would overflow isize."]
        #[doc = ""]
        #[doc = " ### Safety"]
        #[doc = ""]
        #[doc = " The whole of the array must be initialized before it is converted"]
        #[doc = " using [`.assume_init()`] or otherwise traversed/read with the element type `A`."]
        #[doc = ""]
        #[doc = " [`.assume_init()`]: ArrayBase::assume_init"]
        pub fn build_uninit<Sh, F>(shape: Sh, builder: F) -> ArrayBase<S::MaybeUninit, D>
        where
            Sh: ShapeBuilder<Dim = D>,
            F: FnOnce(ArrayViewMut<MaybeUninit<A>, D>),
        {
            let mut array = Self::uninit(shape);
            unsafe {
                builder(array.raw_view_mut_unchecked().deref_into_view_mut());
            }
            array
        }
    }
}
mod impl_methods {
    use crate::argument_traits::AssignElem;
    use crate::dimension;
    use crate::dimension::broadcast::co_broadcast;
    use crate::dimension::reshape_dim;
    use crate::dimension::IntoDimension;
    use crate::dimension::{
        abs_index, axes_of, do_slice, merge_axes, move_min_stride_axis_to_last,
        offset_from_low_addr_ptr_to_logical_ptr, size_of_shape_checked, stride_offset, Axes,
    };
    use crate::error::{self, from_kind, ErrorKind, ShapeError};
    use crate::imp_prelude::*;
    use crate::iter::{
        AxisChunksIter, AxisChunksIterMut, AxisIter, AxisIterMut, ExactChunks, ExactChunksMut,
        IndexedIter, IndexedIterMut, Iter, IterMut, Lanes, LanesMut, Windows,
    };
    use crate::itertools::zip;
    use crate::math_cell::MathCell;
    use crate::order::Order;
    use crate::shape_builder::ShapeArg;
    use crate::slice::{MultiSliceArg, SliceArg};
    use crate::stacking::concatenate;
    use crate::zip::{IntoNdProducer, Zip};
    use crate::AxisDescription;
    use crate::{arraytraits, DimMax};
    use crate::{NdIndex, Slice, SliceInfoElem};
    use alloc::slice;
    use alloc::vec;
    use alloc::vec::Vec;
    use rawpointer::PointerExt;
    use std::mem::{size_of, ManuallyDrop};
    #[doc = " # Methods For All Array Types"]
    impl<A, S, D> ArrayBase<S, D>
    where
        S: RawData<Elem = A>,
        D: Dimension,
    {
        #[doc = " Return the total number of elements in the array."]
        pub fn len(&self) -> usize {
            self.dim.size()
        }
        #[doc = " Return the length of `axis`."]
        #[doc = ""]
        #[doc = " The axis should be in the range `Axis(` 0 .. *n* `)` where *n* is the"]
        #[doc = " number of dimensions (axes) of the array."]
        #[doc = ""]
        #[doc = " ***Panics*** if the axis is out of bounds."]
        pub fn len_of(&self, axis: Axis) -> usize {
            self.dim[axis.index()]
        }
        #[doc = " Return whether the array has any elements"]
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }
        #[doc = " Return the number of dimensions (axes) in the array"]
        pub fn ndim(&self) -> usize {
            self.dim.ndim()
        }
        #[doc = " Return the shape of the array in its “pattern” form,"]
        #[doc = " an integer in the one-dimensional case, tuple in the n-dimensional cases"]
        #[doc = " and so on."]
        pub fn dim(&self) -> D::Pattern {
            self.dim.clone().into_pattern()
        }
        #[doc = " Return the shape of the array as it's stored in the array."]
        #[doc = ""]
        #[doc = " This is primarily useful for passing to other `ArrayBase`"]
        #[doc = " functions, such as when creating another array of the same"]
        #[doc = " shape and dimensionality."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::Array;"]
        #[doc = ""]
        #[doc = " let a = Array::from_elem((2, 3), 5.);"]
        #[doc = ""]
        #[doc = " // Create an array of zeros that's the same shape and dimensionality as `a`."]
        #[doc = " let b = Array::<f64, _>::zeros(a.raw_dim());"]
        #[doc = " ```"]
        pub fn raw_dim(&self) -> D {
            self.dim.clone()
        }
        #[doc = " Return the shape of the array as a slice."]
        #[doc = ""]
        #[doc = " Note that you probably don't want to use this to create an array of the"]
        #[doc = " same shape as another array because creating an array with e.g."]
        #[doc = " [`Array::zeros()`](ArrayBase::zeros) using a shape of type `&[usize]`"]
        #[doc = " results in a dynamic-dimensional array. If you want to create an array"]
        #[doc = " that has the same shape and dimensionality as another array, use"]
        #[doc = " [`.raw_dim()`](ArrayBase::raw_dim) instead:"]
        #[doc = ""]
        #[doc = " ```rust"]
        #[doc = " use ndarray::{Array, Array2};"]
        #[doc = ""]
        #[doc = " let a = Array2::<i32>::zeros((3, 4));"]
        #[doc = " let shape = a.shape();"]
        #[doc = " assert_eq!(shape, &[3, 4]);"]
        #[doc = ""]
        #[doc = " // Since `a.shape()` returned `&[usize]`, we get an `ArrayD` instance:"]
        #[doc = " let b = Array::zeros(shape);"]
        #[doc = " assert_eq!(a.clone().into_dyn(), b);"]
        #[doc = ""]
        #[doc = " // To get the same dimension type, use `.raw_dim()` instead:"]
        #[doc = " let c = Array::zeros(a.raw_dim());"]
        #[doc = " assert_eq!(a, c);"]
        #[doc = " ```"]
        pub fn shape(&self) -> &[usize] {
            self.dim.slice()
        }
        #[doc = " Return the strides of the array as a slice."]
        pub fn strides(&self) -> &[isize] {
            let s = self.strides.slice();
            unsafe { slice::from_raw_parts(s.as_ptr() as *const _, s.len()) }
        }
        #[doc = " Return the stride of `axis`."]
        #[doc = ""]
        #[doc = " The axis should be in the range `Axis(` 0 .. *n* `)` where *n* is the"]
        #[doc = " number of dimensions (axes) of the array."]
        #[doc = ""]
        #[doc = " ***Panics*** if the axis is out of bounds."]
        pub fn stride_of(&self, axis: Axis) -> isize {
            self.strides[axis.index()] as isize
        }
        #[doc = " Return a read-only view of the array"]
        pub fn view(&self) -> ArrayView<'_, A, D>
        where
            S: Data,
        {
            debug_assert!(self.pointer_is_inbounds());
            unsafe { ArrayView::new(self.ptr, self.dim.clone(), self.strides.clone()) }
        }
        #[doc = " Return a read-write view of the array"]
        pub fn view_mut(&mut self) -> ArrayViewMut<'_, A, D>
        where
            S: DataMut,
        {
            self.ensure_unique();
            unsafe { ArrayViewMut::new(self.ptr, self.dim.clone(), self.strides.clone()) }
        }
        #[doc = " Return an uniquely owned copy of the array."]
        #[doc = ""]
        #[doc = " If the input array is contiguous, then the output array will have the same"]
        #[doc = " memory layout. Otherwise, the layout of the output array is unspecified."]
        #[doc = " If you need a particular layout, you can allocate a new array with the"]
        #[doc = " desired memory layout and [`.assign()`](Self::assign) the data."]
        #[doc = " Alternatively, you can collectan iterator, like this for a result in"]
        #[doc = " standard layout:"]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " # use ndarray::prelude::*;"]
        #[doc = " # let arr = Array::from_shape_vec((2, 2).f(), vec![1, 2, 3, 4]).unwrap();"]
        #[doc = " # let owned = {"]
        #[doc = " Array::from_shape_vec(arr.raw_dim(), arr.iter().cloned().collect()).unwrap()"]
        #[doc = " # };"]
        #[doc = " # assert!(owned.is_standard_layout());"]
        #[doc = " # assert_eq!(arr, owned);"]
        #[doc = " ```"]
        #[doc = ""]
        #[doc = " or this for a result in column-major (Fortran) layout:"]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " # use ndarray::prelude::*;"]
        #[doc = " # let arr = Array::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();"]
        #[doc = " # let owned = {"]
        #[doc = " Array::from_shape_vec(arr.raw_dim().f(), arr.t().iter().cloned().collect()).unwrap()"]
        #[doc = " # };"]
        #[doc = " # assert!(owned.t().is_standard_layout());"]
        #[doc = " # assert_eq!(arr, owned);"]
        #[doc = " ```"]
        pub fn to_owned(&self) -> Array<A, D>
        where
            A: Clone,
            S: Data,
        {
            if let Some(slc) = self.as_slice_memory_order() {
                unsafe {
                    Array::from_shape_vec_unchecked(
                        self.dim.clone().strides(self.strides.clone()),
                        slc.to_vec(),
                    )
                }
            } else {
                self.map(A::clone)
            }
        }
        #[doc = " Turn the array into a uniquely owned array, cloning the array elements"]
        #[doc = " if necessary."]
        pub fn into_owned(self) -> Array<A, D>
        where
            A: Clone,
            S: Data,
        {
            S::into_owned(self)
        }
        #[doc = " Turn the array into a shared ownership (copy on write) array,"]
        #[doc = " without any copying."]
        pub fn into_shared(self) -> ArcArray<A, D>
        where
            S: DataOwned,
        {
            let data = self.data.into_shared();
            unsafe {
                ArrayBase::from_data_ptr(data, self.ptr).with_strides_dim(self.strides, self.dim)
            }
        }
        #[doc = " Return an iterator of references to the elements of the array."]
        #[doc = ""]
        #[doc = " Elements are visited in the *logical order* of the array, which"]
        #[doc = " is where the rightmost index is varying the fastest."]
        #[doc = ""]
        #[doc = " Iterator element type is `&A`."]
        pub fn iter(&self) -> Iter<'_, A, D>
        where
            S: Data,
        {
            debug_assert!(self.pointer_is_inbounds());
            self.view().into_iter_()
        }
        #[doc = " Return an iterator of mutable references to the elements of the array."]
        #[doc = ""]
        #[doc = " Elements are visited in the *logical order* of the array, which"]
        #[doc = " is where the rightmost index is varying the fastest."]
        #[doc = ""]
        #[doc = " Iterator element type is `&mut A`."]
        pub fn iter_mut(&mut self) -> IterMut<'_, A, D>
        where
            S: DataMut,
        {
            self.view_mut().into_iter_()
        }
        #[doc = " Slice the array in place along the specified axis."]
        #[doc = ""]
        #[doc = " **Panics** if an index is out of bounds or step size is zero.<br>"]
        #[doc = " **Panics** if `axis` is out of bounds."]
        pub fn slice_axis_inplace(&mut self, axis: Axis, indices: Slice) {
            let offset = do_slice(
                &mut self.dim.slice_mut()[axis.index()],
                &mut self.strides.slice_mut()[axis.index()],
                indices,
            );
            unsafe {
                self.ptr = self.ptr.offset(offset);
            }
            debug_assert!(self.pointer_is_inbounds());
        }
        #[doc = " Return a raw pointer to the element at `index`, or return `None`"]
        #[doc = " if the index is out of bounds."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::arr2;"]
        #[doc = ""]
        #[doc = " let a = arr2(&[[1., 2.], [3., 4.]]);"]
        #[doc = ""]
        #[doc = " let v = a.raw_view();"]
        #[doc = " let p = a.get_ptr((0, 1)).unwrap();"]
        #[doc = ""]
        #[doc = " assert_eq!(unsafe { *p }, 2.);"]
        #[doc = " ```"]
        pub fn get_ptr<I>(&self, index: I) -> Option<*const A>
        where
            I: NdIndex<D>,
        {
            let ptr = self.ptr;
            index
                .index_checked(&self.dim, &self.strides)
                .map(move |offset| unsafe { ptr.as_ptr().offset(offset) as *const _ })
        }
        #[doc = " Return a raw pointer to the element at `index`, or return `None`"]
        #[doc = " if the index is out of bounds."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::arr2;"]
        #[doc = ""]
        #[doc = " let mut a = arr2(&[[1., 2.], [3., 4.]]);"]
        #[doc = ""]
        #[doc = " let v = a.raw_view_mut();"]
        #[doc = " let p = a.get_mut_ptr((0, 1)).unwrap();"]
        #[doc = ""]
        #[doc = " unsafe {"]
        #[doc = "     *p = 5.;"]
        #[doc = " }"]
        #[doc = ""]
        #[doc = " assert_eq!(a.get((0, 1)), Some(&5.));"]
        #[doc = " ```"]
        pub fn get_mut_ptr<I>(&mut self, index: I) -> Option<*mut A>
        where
            S: RawDataMut,
            I: NdIndex<D>,
        {
            let ptr = self.as_mut_ptr();
            index
                .index_checked(&self.dim, &self.strides)
                .map(move |offset| unsafe { ptr.offset(offset) })
        }
        #[doc = " Perform *unchecked* array indexing."]
        #[doc = ""]
        #[doc = " Return a reference to the element at `index`."]
        #[doc = ""]
        #[doc = " **Note:** only unchecked for non-debug builds of ndarray."]
        #[doc = ""]
        #[doc = " # Safety"]
        #[doc = ""]
        #[doc = " The caller must ensure that the index is in-bounds."]
        #[inline]
        pub unsafe fn uget<I>(&self, index: I) -> &A
        where
            S: Data,
            I: NdIndex<D>,
        {
            arraytraits::debug_bounds_check(self, &index);
            let off = index.index_unchecked(&self.strides);
            &*self.ptr.as_ptr().offset(off)
        }
        fn get_0d(&self) -> &A
        where
            S: Data,
        {
            assert!(self.ndim() == 0);
            unsafe { &*self.as_ptr() }
        }
        #[doc = " Returns a view restricted to `index` along the axis, with the axis"]
        #[doc = " removed."]
        #[doc = ""]
        #[doc = " See [*Subviews*](#subviews) for full documentation."]
        #[doc = ""]
        #[doc = " **Panics** if `axis` or `index` is out of bounds."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::{arr2, ArrayView, Axis};"]
        #[doc = ""]
        #[doc = " let a = arr2(&[[1., 2. ],    // ... axis 0, row 0"]
        #[doc = "                [3., 4. ],    // --- axis 0, row 1"]
        #[doc = "                [5., 6. ]]);  // ... axis 0, row 2"]
        #[doc = " //               .   \\"]
        #[doc = " //                .   axis 1, column 1"]
        #[doc = " //                 axis 1, column 0"]
        #[doc = " assert!("]
        #[doc = "     a.index_axis(Axis(0), 1) == ArrayView::from(&[3., 4.]) &&"]
        #[doc = "     a.index_axis(Axis(1), 1) == ArrayView::from(&[2., 4., 6.])"]
        #[doc = " );"]
        #[doc = " ```"]
        pub fn index_axis(&self, axis: Axis, index: usize) -> ArrayView<'_, A, D::Smaller>
        where
            S: Data,
            D: RemoveAxis,
        {
            self.view().index_axis_move(axis, index)
        }
        #[doc = " Returns a mutable view restricted to `index` along the axis, with the"]
        #[doc = " axis removed."]
        #[doc = ""]
        #[doc = " **Panics** if `axis` or `index` is out of bounds."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::{arr2, aview2, Axis};"]
        #[doc = ""]
        #[doc = " let mut a = arr2(&[[1., 2. ],"]
        #[doc = "                    [3., 4. ]]);"]
        #[doc = " //                   .   \\"]
        #[doc = " //                    .   axis 1, column 1"]
        #[doc = " //                     axis 1, column 0"]
        #[doc = ""]
        #[doc = " {"]
        #[doc = "     let mut column1 = a.index_axis_mut(Axis(1), 1);"]
        #[doc = "     column1 += 10.;"]
        #[doc = " }"]
        #[doc = ""]
        #[doc = " assert!("]
        #[doc = "     a == aview2(&[[1., 12.],"]
        #[doc = "                   [3., 14.]])"]
        #[doc = " );"]
        #[doc = " ```"]
        pub fn index_axis_mut(
            &mut self,
            axis: Axis,
            index: usize,
        ) -> ArrayViewMut<'_, A, D::Smaller>
        where
            S: DataMut,
            D: RemoveAxis,
        {
            self.view_mut().index_axis_move(axis, index)
        }
        #[doc = " Collapses the array to `index` along the axis and removes the axis."]
        #[doc = ""]
        #[doc = " See [`.index_axis()`](Self::index_axis) and [*Subviews*](#subviews) for full documentation."]
        #[doc = ""]
        #[doc = " **Panics** if `axis` or `index` is out of bounds."]
        pub fn index_axis_move(mut self, axis: Axis, index: usize) -> ArrayBase<S, D::Smaller>
        where
            D: RemoveAxis,
        {
            self.collapse_axis(axis, index);
            let dim = self.dim.remove_axis(axis);
            let strides = self.strides.remove_axis(axis);
            unsafe { self.with_strides_dim(strides, dim) }
        }
        #[doc = " Selects `index` along the axis, collapsing the axis into length one."]
        #[doc = ""]
        #[doc = " **Panics** if `axis` or `index` is out of bounds."]
        pub fn collapse_axis(&mut self, axis: Axis, index: usize) {
            let offset =
                dimension::do_collapse_axis(&mut self.dim, &self.strides, axis.index(), index);
            self.ptr = unsafe { self.ptr.offset(offset) };
            debug_assert!(self.pointer_is_inbounds());
        }
        #[doc = " Return a producer and iterable that traverses over the *generalized*"]
        #[doc = " rows of the array. For a 2D array these are the regular rows."]
        #[doc = ""]
        #[doc = " This is equivalent to `.lanes(Axis(n - 1))` where *n* is `self.ndim()`."]
        #[doc = ""]
        #[doc = " For an array of dimensions *a* × *b* × *c* × ... × *l* × *m*"]
        #[doc = " it has *a* × *b* × *c* × ... × *l* rows each of length *m*."]
        #[doc = ""]
        #[doc = " For example, in a 2 × 2 × 3 array, each row is 3 elements long"]
        #[doc = " and there are 2 × 2 = 4 rows in total."]
        #[doc = ""]
        #[doc = " Iterator element is `ArrayView1<A>` (1D array view)."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::arr3;"]
        #[doc = ""]
        #[doc = " let a = arr3(&[[[ 0,  1,  2],    // -- row 0, 0"]
        #[doc = "                 [ 3,  4,  5]],   // -- row 0, 1"]
        #[doc = "                [[ 6,  7,  8],    // -- row 1, 0"]
        #[doc = "                 [ 9, 10, 11]]]); // -- row 1, 1"]
        #[doc = ""]
        #[doc = " // `rows` will yield the four generalized rows of the array."]
        #[doc = " for row in a.rows() {"]
        #[doc = "     /* loop body */"]
        #[doc = " }"]
        #[doc = " ```"]
        pub fn rows(&self) -> Lanes<'_, A, D::Smaller>
        where
            S: Data,
        {
            let mut n = self.ndim();
            if n == 0 {
                n += 1;
            }
            Lanes::new(self.view(), Axis(n - 1))
        }
        #[doc = " Return a producer and iterable that traverses over the *generalized*"]
        #[doc = " rows of the array and yields mutable array views."]
        #[doc = ""]
        #[doc = " Iterator element is `ArrayView1<A>` (1D read-write array view)."]
        pub fn rows_mut(&mut self) -> LanesMut<'_, A, D::Smaller>
        where
            S: DataMut,
        {
            let mut n = self.ndim();
            if n == 0 {
                n += 1;
            }
            LanesMut::new(self.view_mut(), Axis(n - 1))
        }
        #[doc = " Return a producer and iterable that traverses over the *generalized*"]
        #[doc = " columns of the array. For a 2D array these are the regular columns."]
        #[doc = ""]
        #[doc = " This is equivalent to `.lanes(Axis(0))`."]
        #[doc = ""]
        #[doc = " For an array of dimensions *a* × *b* × *c* × ... × *l* × *m*"]
        #[doc = " it has *b* × *c* × ... × *l* × *m* columns each of length *a*."]
        #[doc = ""]
        #[doc = " For example, in a 2 × 2 × 3 array, each column is 2 elements long"]
        #[doc = " and there are 2 × 3 = 6 columns in total."]
        #[doc = ""]
        #[doc = " Iterator element is `ArrayView1<A>` (1D array view)."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::arr3;"]
        #[doc = ""]
        #[doc = " // The generalized columns of a 3D array:"]
        #[doc = " // are directed along the 0th axis: 0 and 6, 1 and 7 and so on..."]
        #[doc = " let a = arr3(&[[[ 0,  1,  2], [ 3,  4,  5]],"]
        #[doc = "                [[ 6,  7,  8], [ 9, 10, 11]]]);"]
        #[doc = ""]
        #[doc = " // Here `columns` will yield the six generalized columns of the array."]
        #[doc = " for column in a.columns() {"]
        #[doc = "     /* loop body */"]
        #[doc = " }"]
        #[doc = " ```"]
        pub fn columns(&self) -> Lanes<'_, A, D::Smaller>
        where
            S: Data,
        {
            Lanes::new(self.view(), Axis(0))
        }
        #[doc = " Return a producer and iterable that traverses over the *generalized*"]
        #[doc = " columns of the array and yields mutable array views."]
        #[doc = ""]
        #[doc = " Iterator element is `ArrayView1<A>` (1D read-write array view)."]
        pub fn columns_mut(&mut self) -> LanesMut<'_, A, D::Smaller>
        where
            S: DataMut,
        {
            LanesMut::new(self.view_mut(), Axis(0))
        }
        #[doc = " Return a producer and iterable that traverses over all 1D lanes"]
        #[doc = " pointing in the direction of `axis`."]
        #[doc = ""]
        #[doc = " Iterator element is `ArrayViewMut1<A>` (1D read-write array view)."]
        pub fn lanes_mut(&mut self, axis: Axis) -> LanesMut<'_, A, D::Smaller>
        where
            S: DataMut,
        {
            LanesMut::new(self.view_mut(), axis)
        }
        #[doc = " Return an iterator that traverses over `axis`"]
        #[doc = " and yields each subview along it."]
        #[doc = ""]
        #[doc = " For example, in a 3 × 4 × 5 array, with `axis` equal to `Axis(2)`,"]
        #[doc = " the iterator element"]
        #[doc = " is a 3 × 4 subview (and there are 5 in total), as shown"]
        #[doc = " in the picture below."]
        #[doc = ""]
        #[doc = " Iterator element is `ArrayView<A, D::Smaller>` (read-only array view)."]
        #[doc = ""]
        #[doc = " See [*Subviews*](#subviews) for full documentation."]
        #[doc = ""]
        #[doc = " **Panics** if `axis` is out of bounds."]
        #[doc = ""]
        #[doc = " <img src=\"https://rust-ndarray.github.io/ndarray/images/axis_iter_3_4_5.svg\" height=\"250px\">"]
        pub fn axis_iter(&self, axis: Axis) -> AxisIter<'_, A, D::Smaller>
        where
            S: Data,
            D: RemoveAxis,
        {
            AxisIter::new(self.view(), axis)
        }
        fn diag_params(&self) -> (Ix, Ixs) {
            let len = self.dim.slice().iter().cloned().min().unwrap_or(1);
            let stride = self.strides().iter().sum();
            (len, stride)
        }
        #[doc = " Try to make the array unshared."]
        #[doc = ""]
        #[doc = " This is equivalent to `.ensure_unique()` if `S: DataMut`."]
        #[doc = ""]
        #[doc = " This method is mostly only useful with unsafe code."]
        fn try_ensure_unique(&mut self)
        where
            S: RawDataMut,
        {
            debug_assert!(self.pointer_is_inbounds());
            S::try_ensure_unique(self);
            debug_assert!(self.pointer_is_inbounds());
        }
        #[doc = " Make the array unshared."]
        #[doc = ""]
        #[doc = " This method is mostly only useful with unsafe code."]
        fn ensure_unique(&mut self)
        where
            S: DataMut,
        {
            debug_assert!(self.pointer_is_inbounds());
            S::ensure_unique(self);
            debug_assert!(self.pointer_is_inbounds());
        }
        #[doc = " Return `true` if the array data is laid out in contiguous “C order” in"]
        #[doc = " memory (where the last index is the most rapidly varying)."]
        #[doc = ""]
        #[doc = " Return `false` otherwise, i.e. the array is possibly not"]
        #[doc = " contiguous in memory, it has custom strides, etc."]
        pub fn is_standard_layout(&self) -> bool {
            dimension::is_layout_c(&self.dim, &self.strides)
        }
        #[doc = " Return true if the array is known to be contiguous."]
        pub(crate) fn is_contiguous(&self) -> bool {
            D::is_contiguous(&self.dim, &self.strides)
        }
        #[doc = " Return a pointer to the first element in the array."]
        #[doc = ""]
        #[doc = " Raw access to array elements needs to follow the strided indexing"]
        #[doc = " scheme: an element at multi-index *I* in an array with strides *S* is"]
        #[doc = " located at offset"]
        #[doc = ""]
        #[doc = " *Σ<sub>0 ≤ k < d</sub> I<sub>k</sub> × S<sub>k</sub>*"]
        #[doc = ""]
        #[doc = " where *d* is `self.ndim()`."]
        #[inline(always)]
        pub fn as_ptr(&self) -> *const A {
            self.ptr.as_ptr() as *const A
        }
        #[doc = " Return a mutable pointer to the first element in the array."]
        #[doc = ""]
        #[doc = " This method attempts to unshare the data. If `S: DataMut`, then the"]
        #[doc = " data is guaranteed to be uniquely held on return."]
        #[doc = ""]
        #[doc = " # Warning"]
        #[doc = ""]
        #[doc = " When accessing elements through this pointer, make sure to use strides"]
        #[doc = " obtained *after* calling this method, since the process of unsharing"]
        #[doc = " the data may change the strides."]
        #[inline(always)]
        pub fn as_mut_ptr(&mut self) -> *mut A
        where
            S: RawDataMut,
        {
            self.try_ensure_unique();
            self.ptr.as_ptr()
        }
        #[doc = " Return a raw view of the array."]
        #[inline]
        pub fn raw_view(&self) -> RawArrayView<A, D> {
            unsafe { RawArrayView::new(self.ptr, self.dim.clone(), self.strides.clone()) }
        }
        #[doc = " Return a raw mutable view of the array."]
        #[doc = ""]
        #[doc = " This method attempts to unshare the data. If `S: DataMut`, then the"]
        #[doc = " data is guaranteed to be uniquely held on return."]
        #[inline]
        pub fn raw_view_mut(&mut self) -> RawArrayViewMut<A, D>
        where
            S: RawDataMut,
        {
            self.try_ensure_unique();
            unsafe { RawArrayViewMut::new(self.ptr, self.dim.clone(), self.strides.clone()) }
        }
        #[doc = " Return a raw mutable view of the array."]
        #[doc = ""]
        #[doc = " Safety: The caller must ensure that the owned array is unshared when this is called"]
        #[inline]
        pub(crate) unsafe fn raw_view_mut_unchecked(&mut self) -> RawArrayViewMut<A, D>
        where
            S: DataOwned,
        {
            RawArrayViewMut::new(self.ptr, self.dim.clone(), self.strides.clone())
        }
        #[doc = " Return the array’s data as a slice, if it is contiguous and in standard order."]
        #[doc = " Return `None` otherwise."]
        #[doc = ""]
        #[doc = " If this function returns `Some(_)`, then the element order in the slice"]
        #[doc = " corresponds to the logical order of the array’s elements."]
        pub fn as_slice(&self) -> Option<&[A]>
        where
            S: Data,
        {
            if self.is_standard_layout() {
                unsafe { Some(slice::from_raw_parts(self.ptr.as_ptr(), self.len())) }
            } else {
                None
            }
        }
        #[doc = " Return the array’s data as a slice if it is contiguous,"]
        #[doc = " return `None` otherwise."]
        #[doc = ""]
        #[doc = " If this function returns `Some(_)`, then the elements in the slice"]
        #[doc = " have whatever order the elements have in memory."]
        pub fn as_slice_memory_order(&self) -> Option<&[A]>
        where
            S: Data,
        {
            if self.is_contiguous() {
                let offset = offset_from_low_addr_ptr_to_logical_ptr(&self.dim, &self.strides);
                unsafe {
                    Some(slice::from_raw_parts(
                        self.ptr.sub(offset).as_ptr(),
                        self.len(),
                    ))
                }
            } else {
                None
            }
        }
        #[doc = " Return the array’s data as a slice if it is contiguous,"]
        #[doc = " return `None` otherwise."]
        #[doc = ""]
        #[doc = " In the contiguous case, in order to return a unique reference, this"]
        #[doc = " method unshares the data if necessary, but it preserves the existing"]
        #[doc = " strides."]
        pub fn as_slice_memory_order_mut(&mut self) -> Option<&mut [A]>
        where
            S: DataMut,
        {
            self.try_as_slice_memory_order_mut().ok()
        }
        #[doc = " Return the array’s data as a slice if it is contiguous, otherwise"]
        #[doc = " return `self` in the `Err` variant."]
        pub(crate) fn try_as_slice_memory_order_mut(&mut self) -> Result<&mut [A], &mut Self>
        where
            S: DataMut,
        {
            if self.is_contiguous() {
                self.ensure_unique();
                let offset = offset_from_low_addr_ptr_to_logical_ptr(&self.dim, &self.strides);
                unsafe {
                    Ok(slice::from_raw_parts_mut(
                        self.ptr.sub(offset).as_ptr(),
                        self.len(),
                    ))
                }
            } else {
                Err(self)
            }
        }
        #[doc = " Convert an array or array view to another with the same type, but different dimensionality"]
        #[doc = " type. Errors if the dimensions don't agree (the number of axes must match)."]
        #[doc = ""]
        #[doc = " Note that conversion to a dynamic dimensional array will never fail (and is equivalent to"]
        #[doc = " the `into_dyn` method)."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::{ArrayD, Ix2, IxDyn};"]
        #[doc = ""]
        #[doc = " // Create a dynamic dimensionality array and convert it to an Array2"]
        #[doc = " // (Ix2 dimension type)."]
        #[doc = ""]
        #[doc = " let array = ArrayD::<f64>::zeros(IxDyn(&[10, 10]));"]
        #[doc = ""]
        #[doc = " assert!(array.into_dimensionality::<Ix2>().is_ok());"]
        #[doc = " ```"]
        pub fn into_dimensionality<D2>(self) -> Result<ArrayBase<S, D2>, ShapeError>
        where
            D2: Dimension,
        {
            unsafe {
                if D::NDIM == D2::NDIM {
                    let dim = unlimited_transmute::<D, D2>(self.dim);
                    let strides = unlimited_transmute::<D, D2>(self.strides);
                    return Ok(ArrayBase::from_data_ptr(self.data, self.ptr)
                        .with_strides_dim(strides, dim));
                } else if D::NDIM == None || D2::NDIM == None {
                    if let Some(dim) = D2::from_dimension(&self.dim) {
                        if let Some(strides) = D2::from_dimension(&self.strides) {
                            return Ok(self.with_strides_dim(strides, dim));
                        }
                    }
                }
            }
            Err(ShapeError::from_kind(ErrorKind::IncompatibleShape))
        }
        #[doc = " Act like a larger size and/or shape array by *broadcasting*"]
        #[doc = " into a larger shape, if possible."]
        #[doc = ""]
        #[doc = " Return `None` if shapes can not be broadcast together."]
        #[doc = ""]
        #[doc = " ***Background***"]
        #[doc = ""]
        #[doc = "  * Two axes are compatible if they are equal, or one of them is 1."]
        #[doc = "  * In this instance, only the axes of the smaller side (self) can be 1."]
        #[doc = ""]
        #[doc = " Compare axes beginning with the *last* axis of each shape."]
        #[doc = ""]
        #[doc = " For example (1, 2, 4) can be broadcast into (7, 6, 2, 4)"]
        #[doc = " because its axes are either equal or 1 (or missing);"]
        #[doc = " while (2, 2) can *not* be broadcast into (2, 4)."]
        #[doc = ""]
        #[doc = " The implementation creates a view with strides set to zero for the"]
        #[doc = " axes that are to be repeated."]
        #[doc = ""]
        #[doc = " The broadcasting documentation for Numpy has more information."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::{aview1, aview2};"]
        #[doc = ""]
        #[doc = " assert!("]
        #[doc = "     aview1(&[1., 0.]).broadcast((10, 2)).unwrap()"]
        #[doc = "     == aview2(&[[1., 0.]; 10])"]
        #[doc = " );"]
        #[doc = " ```"]
        pub fn broadcast<E>(&self, dim: E) -> Option<ArrayView<'_, A, E::Dim>>
        where
            E: IntoDimension,
            S: Data,
        {
            #[doc = " Return new stride when trying to grow `from` into shape `to`"]
            #[doc = ""]
            #[doc = " Broadcasting works by returning a \"fake stride\" where elements"]
            #[doc = " to repeat are in axes with 0 stride, so that several indexes point"]
            #[doc = " to the same element."]
            #[doc = ""]
            #[doc = " **Note:** Cannot be used for mutable iterators, since repeating"]
            #[doc = " elements would create aliasing pointers."]
            fn upcast<D: Dimension, E: Dimension>(to: &D, from: &E, stride: &E) -> Option<D> {
                let _ = size_of_shape_checked(to).ok()?;
                let mut new_stride = to.clone();
                if to.ndim() < from.ndim() {
                    return None;
                }
                {
                    let mut new_stride_iter = new_stride.slice_mut().iter_mut().rev();
                    for ((er, es), dr) in from
                        .slice()
                        .iter()
                        .rev()
                        .zip(stride.slice().iter().rev())
                        .zip(new_stride_iter.by_ref())
                    {
                        if *dr == *er {
                            *dr = *es;
                        } else if *er == 1 {
                            *dr = 0
                        } else {
                            return None;
                        }
                    }
                    for dr in new_stride_iter {
                        *dr = 0;
                    }
                }
                Some(new_stride)
            }
            let dim = dim.into_dimension();
            let broadcast_strides = match upcast(&dim, &self.dim, &self.strides) {
                Some(st) => st,
                None => return None,
            };
            unsafe { Some(ArrayView::new(self.ptr, dim, broadcast_strides)) }
        }
        #[doc = " Transpose the array by reversing axes."]
        #[doc = ""]
        #[doc = " Transposition reverses the order of the axes (dimensions and strides)"]
        #[doc = " while retaining the same data."]
        pub fn reversed_axes(mut self) -> ArrayBase<S, D> {
            self.dim.slice_mut().reverse();
            self.strides.slice_mut().reverse();
            self
        }
        #[doc = " Return an iterator over the length and stride of each axis."]
        pub fn axes(&self) -> Axes<'_, D> {
            axes_of(&self.dim, &self.strides)
        }
        #[doc = " Reverse the stride of `axis`."]
        #[doc = ""]
        #[doc = " ***Panics*** if the axis is out of bounds."]
        pub fn invert_axis(&mut self, axis: Axis) {
            unsafe {
                let s = self.strides.axis(axis) as Ixs;
                let m = self.dim.axis(axis);
                if m != 0 {
                    self.ptr = self.ptr.offset(stride_offset(m - 1, s as Ix));
                }
                self.strides.set_axis(axis, (-s) as Ix);
            }
        }
        #[doc = " Insert new array axis at `axis` and return the result."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::{Array3, Axis, arr1, arr2};"]
        #[doc = ""]
        #[doc = " // Convert a 1-D array into a row vector (2-D)."]
        #[doc = " let a = arr1(&[1, 2, 3]);"]
        #[doc = " let row = a.insert_axis(Axis(0));"]
        #[doc = " assert_eq!(row, arr2(&[[1, 2, 3]]));"]
        #[doc = ""]
        #[doc = " // Convert a 1-D array into a column vector (2-D)."]
        #[doc = " let b = arr1(&[1, 2, 3]);"]
        #[doc = " let col = b.insert_axis(Axis(1));"]
        #[doc = " assert_eq!(col, arr2(&[[1], [2], [3]]));"]
        #[doc = ""]
        #[doc = " // The new axis always has length 1."]
        #[doc = " let b = Array3::<f64>::zeros((3, 4, 5));"]
        #[doc = " assert_eq!(b.insert_axis(Axis(2)).shape(), &[3, 4, 1, 5]);"]
        #[doc = " ```"]
        #[doc = ""]
        #[doc = " ***Panics*** if the axis is out of bounds."]
        pub fn insert_axis(self, axis: Axis) -> ArrayBase<S, D::Larger> {
            assert!(axis.index() <= self.ndim());
            unsafe {
                let strides = self.strides.insert_axis(axis);
                let dim = self.dim.insert_axis(axis);
                self.with_strides_dim(strides, dim)
            }
        }
        pub(crate) fn pointer_is_inbounds(&self) -> bool {
            self.data._is_pointer_inbounds(self.as_ptr())
        }
        pub(crate) fn zip_mut_with_same_shape<B, S2, E, F>(
            &mut self,
            rhs: &ArrayBase<S2, E>,
            mut f: F,
        ) where
            S: DataMut,
            S2: Data<Elem = B>,
            E: Dimension,
            F: FnMut(&mut A, &B),
        {
            debug_assert_eq!(self.shape(), rhs.shape());
            if self.dim.strides_equivalent(&self.strides, &rhs.strides) {
                if let Some(self_s) = self.as_slice_memory_order_mut() {
                    if let Some(rhs_s) = rhs.as_slice_memory_order() {
                        for (s, r) in self_s.iter_mut().zip(rhs_s) {
                            f(s, r);
                        }
                        return;
                    }
                }
            }
            self.zip_mut_with_by_rows(rhs, f);
        }
        #[inline(always)]
        fn zip_mut_with_by_rows<B, S2, E, F>(&mut self, rhs: &ArrayBase<S2, E>, mut f: F)
        where
            S: DataMut,
            S2: Data<Elem = B>,
            E: Dimension,
            F: FnMut(&mut A, &B),
        {
            debug_assert_eq!(self.shape(), rhs.shape());
            debug_assert_ne!(self.ndim(), 0);
            let n = self.ndim();
            let dim = self.raw_dim();
            Zip::from(LanesMut::new(self.view_mut(), Axis(n - 1)))
                .and(Lanes::new(rhs.broadcast_assume(dim), Axis(n - 1)))
                .for_each(move |s_row, r_row| Zip::from(s_row).and(r_row).for_each(|a, b| f(a, b)));
        }
        fn zip_mut_with_elem<B, F>(&mut self, rhs_elem: &B, mut f: F)
        where
            S: DataMut,
            F: FnMut(&mut A, &B),
        {
            self.map_inplace(move |elt| f(elt, rhs_elem));
        }
        #[doc = " Traverse two arrays in unspecified order, in lock step,"]
        #[doc = " calling the closure `f` on each element pair."]
        #[doc = ""]
        #[doc = " If their shapes disagree, `rhs` is broadcast to the shape of `self`."]
        #[doc = ""]
        #[doc = " **Panics** if broadcasting isn’t possible."]
        #[inline]
        pub fn zip_mut_with<B, S2, E, F>(&mut self, rhs: &ArrayBase<S2, E>, f: F)
        where
            S: DataMut,
            S2: Data<Elem = B>,
            E: Dimension,
            F: FnMut(&mut A, &B),
        {
            if rhs.dim.ndim() == 0 {
                self.zip_mut_with_elem(rhs.get_0d(), f);
            } else if self.dim.ndim() == rhs.dim.ndim() && self.shape() == rhs.shape() {
                self.zip_mut_with_same_shape(rhs, f);
            } else {
                let rhs_broadcast = rhs.broadcast_unwrap(self.raw_dim());
                self.zip_mut_with_by_rows(&rhs_broadcast, f);
            }
        }
        #[doc = " Traverse the array elements and apply a fold,"]
        #[doc = " returning the resulting value."]
        #[doc = ""]
        #[doc = " Elements are visited in arbitrary order."]
        pub fn fold<'a, F, B>(&'a self, init: B, f: F) -> B
        where
            F: FnMut(B, &'a A) -> B,
            A: 'a,
            S: Data,
        {
            if let Some(slc) = self.as_slice_memory_order() {
                slc.iter().fold(init, f)
            } else {
                let mut v = self.view();
                move_min_stride_axis_to_last(&mut v.dim, &mut v.strides);
                v.into_elements_base().fold(init, f)
            }
        }
        #[doc = " Call `f` by reference on each element and create a new array"]
        #[doc = " with the new values."]
        #[doc = ""]
        #[doc = " Elements are visited in arbitrary order."]
        #[doc = ""]
        #[doc = " Return an array with the same shape as `self`."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::arr2;"]
        #[doc = ""]
        #[doc = " let a = arr2(&[[ 0., 1.],"]
        #[doc = "                [-1., 2.]]);"]
        #[doc = " assert!("]
        #[doc = "     a.map(|x| *x >= 1.0)"]
        #[doc = "     == arr2(&[[false, true],"]
        #[doc = "               [false, true]])"]
        #[doc = " );"]
        #[doc = " ```"]
        pub fn map<'a, B, F>(&'a self, f: F) -> Array<B, D>
        where
            F: FnMut(&'a A) -> B,
            A: 'a,
            S: Data,
        {
            unsafe {
                if let Some(slc) = self.as_slice_memory_order() {
                    ArrayBase::from_shape_trusted_iter_unchecked(
                        self.dim.clone().strides(self.strides.clone()),
                        slc.iter(),
                        f,
                    )
                } else {
                    ArrayBase::from_shape_trusted_iter_unchecked(self.dim.clone(), self.iter(), f)
                }
            }
        }
        #[doc = " Call `f` on a mutable reference of each element and create a new array"]
        #[doc = " with the new values."]
        #[doc = ""]
        #[doc = " Elements are visited in arbitrary order."]
        #[doc = ""]
        #[doc = " Return an array with the same shape as `self`."]
        pub fn map_mut<'a, B, F>(&'a mut self, f: F) -> Array<B, D>
        where
            F: FnMut(&'a mut A) -> B,
            A: 'a,
            S: DataMut,
        {
            let dim = self.dim.clone();
            if self.is_contiguous() {
                let strides = self.strides.clone();
                let slc = self.as_slice_memory_order_mut().unwrap();
                unsafe {
                    ArrayBase::from_shape_trusted_iter_unchecked(
                        dim.strides(strides),
                        slc.iter_mut(),
                        f,
                    )
                }
            } else {
                unsafe { ArrayBase::from_shape_trusted_iter_unchecked(dim, self.iter_mut(), f) }
            }
        }
        #[doc = " Call `f` by **v**alue on each element and create a new array"]
        #[doc = " with the new values."]
        #[doc = ""]
        #[doc = " Elements are visited in arbitrary order."]
        #[doc = ""]
        #[doc = " Return an array with the same shape as `self`."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::arr2;"]
        #[doc = ""]
        #[doc = " let a = arr2(&[[ 0., 1.],"]
        #[doc = "                [-1., 2.]]);"]
        #[doc = " assert!("]
        #[doc = "     a.mapv(f32::abs) == arr2(&[[0., 1.],"]
        #[doc = "                                [1., 2.]])"]
        #[doc = " );"]
        #[doc = " ```"]
        pub fn mapv<B, F>(&self, mut f: F) -> Array<B, D>
        where
            F: FnMut(A) -> B,
            A: Clone,
            S: Data,
        {
            self.map(move |x| f(x.clone()))
        }
        #[doc = " Call `f` by **v**alue on each element, update the array with the new values"]
        #[doc = " and return it."]
        #[doc = ""]
        #[doc = " Elements are visited in arbitrary order."]
        pub fn mapv_into<F>(mut self, f: F) -> Self
        where
            S: DataMut,
            F: FnMut(A) -> A,
            A: Clone,
        {
            self.mapv_inplace(f);
            self
        }
        #[doc = " Modify the array in place by calling `f` by mutable reference on each element."]
        #[doc = ""]
        #[doc = " Elements are visited in arbitrary order."]
        pub fn map_inplace<'a, F>(&'a mut self, f: F)
        where
            S: DataMut,
            A: 'a,
            F: FnMut(&'a mut A),
        {
            match self.try_as_slice_memory_order_mut() {
                Ok(slc) => slc.iter_mut().for_each(f),
                Err(arr) => {
                    let mut v = arr.view_mut();
                    move_min_stride_axis_to_last(&mut v.dim, &mut v.strides);
                    v.into_elements_base().for_each(f);
                }
            }
        }
        #[doc = " Modify the array in place by calling `f` by **v**alue on each element."]
        #[doc = " The array is updated with the new values."]
        #[doc = ""]
        #[doc = " Elements are visited in arbitrary order."]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " # #[cfg(feature = \"approx\")] {"]
        #[doc = " use approx::assert_abs_diff_eq;"]
        #[doc = " use ndarray::arr2;"]
        #[doc = ""]
        #[doc = " let mut a = arr2(&[[ 0., 1.],"]
        #[doc = "                    [-1., 2.]]);"]
        #[doc = " a.mapv_inplace(f32::exp);"]
        #[doc = " assert_abs_diff_eq!("]
        #[doc = "     a,"]
        #[doc = "     arr2(&[[1.00000, 2.71828],"]
        #[doc = "            [0.36788, 7.38906]]),"]
        #[doc = "     epsilon = 1e-5,"]
        #[doc = " );"]
        #[doc = " # }"]
        #[doc = " ```"]
        pub fn mapv_inplace<F>(&mut self, mut f: F)
        where
            S: DataMut,
            F: FnMut(A) -> A,
            A: Clone,
        {
            self.map_inplace(move |x| *x = f(x.clone()));
        }
        #[doc = " Call `f` for each element in the array."]
        #[doc = ""]
        #[doc = " Elements are visited in arbitrary order."]
        pub fn for_each<'a, F>(&'a self, mut f: F)
        where
            F: FnMut(&'a A),
            A: 'a,
            S: Data,
        {
            self.fold((), move |(), elt| f(elt))
        }
    }
    #[doc = " Transmute from A to B."]
    #[doc = ""]
    #[doc = " Like transmute, but does not have the compile-time size check which blocks"]
    #[doc = " using regular transmute in some cases."]
    #[doc = ""]
    #[doc = " **Panics** if the size of A and B are different."]
    #[inline]
    unsafe fn unlimited_transmute<A, B>(data: A) -> B {
        assert_eq!(size_of::<A>(), size_of::<B>());
        let old_data = ManuallyDrop::new(data);
        (&*old_data as *const A as *const B).read()
    }
    type DimMaxOf<A, B> = <A as DimMax<B>>::Output;
}
mod impl_owned_array {
    use crate::dimension;
    use crate::error::{ErrorKind, ShapeError};
    use crate::imp_prelude::*;
    use crate::iterators::Baseiter;
    use crate::low_level_util::AbortIfPanic;
    use crate::OwnedRepr;
    use crate::Zip;
    use alloc::vec::Vec;
    use rawpointer::PointerExt;
    use std::mem;
    use std::mem::MaybeUninit;
    impl<A, D> Array<A, D>
    where
        D: Dimension,
    {
        #[doc = " Move all elements from self into `new_array`, which must be of the same shape but"]
        #[doc = " can have a different memory layout. The destination is overwritten completely."]
        #[doc = ""]
        #[doc = " The destination should be a mut reference to an array or an `ArrayViewMut` with"]
        #[doc = " `MaybeUninit<A>` elements (which are overwritten without dropping any existing value)."]
        #[doc = ""]
        #[doc = " Minor implementation note: Owned arrays like `self` may be sliced in place and own elements"]
        #[doc = " that are not part of their active view; these are dropped at the end of this function,"]
        #[doc = " after all elements in the \"active view\" are moved into `new_array`. If there is a panic in"]
        #[doc = " drop of any such element, other elements may be leaked."]
        #[doc = ""]
        #[doc = " ***Panics*** if the shapes don't agree."]
        #[doc = ""]
        #[doc = " ## Example"]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::Array;"]
        #[doc = ""]
        #[doc = " let a = Array::from_iter(0..100).into_shape((10, 10)).unwrap();"]
        #[doc = " let mut b = Array::uninit((10, 10));"]
        #[doc = " a.move_into_uninit(&mut b);"]
        #[doc = " unsafe {"]
        #[doc = "     // we can now promise we have fully initialized `b`."]
        #[doc = "     let b = b.assume_init();"]
        #[doc = " }"]
        #[doc = " ```"]
        pub fn move_into_uninit<'a, AM>(self, new_array: AM)
        where
            AM: Into<ArrayViewMut<'a, MaybeUninit<A>, D>>,
            A: 'a,
        {
            self.move_into_impl(new_array.into())
        }
        fn move_into_impl(mut self, new_array: ArrayViewMut<MaybeUninit<A>, D>) {
            unsafe {
                let guard = AbortIfPanic(&"move_into: moving out of owned value");
                Zip::from(self.raw_view_mut())
                    .and(new_array)
                    .for_each(|src, dst| {
                        src.copy_to_nonoverlapping(dst.as_mut_ptr(), 1);
                    });
                guard.defuse();
                self.drop_unreachable_elements();
            }
        }
        #[doc = " This drops all \"unreachable\" elements in the data storage of self."]
        #[doc = ""]
        #[doc = " That means those elements that are not visible in the slicing of the array."]
        #[doc = " *Reachable elements are assumed to already have been moved from.*"]
        #[doc = ""]
        #[doc = " # Safety"]
        #[doc = ""]
        #[doc = " This is a panic critical section since `self` is already moved-from."]
        fn drop_unreachable_elements(mut self) -> OwnedRepr<A> {
            let self_len = self.len();
            let data_len = self.data.len();
            let has_unreachable_elements = self_len != data_len;
            if !has_unreachable_elements || mem::size_of::<A>() == 0 || !mem::needs_drop::<A>() {
                unsafe {
                    self.data.set_len(0);
                }
                self.data
            } else {
                self.drop_unreachable_elements_slow()
            }
        }
        #[inline(never)]
        #[cold]
        fn drop_unreachable_elements_slow(mut self) -> OwnedRepr<A> {
            let data_len = self.data.len();
            let data_ptr = self.data.as_nonnull_mut().as_ptr();
            unsafe {
                let self_ = self.raw_view_mut();
                self.data.set_len(0);
                drop_unreachable_raw(self_, data_ptr, data_len);
            }
            self.data
        }
        #[doc = " Create an empty array with an all-zeros shape"]
        #[doc = ""]
        #[doc = " ***Panics*** if D is zero-dimensional, because it can't be empty"]
        pub(crate) fn empty() -> Array<A, D> {
            assert_ne!(D::NDIM, Some(0));
            let ndim = D::NDIM.unwrap_or(1);
            Array::from_shape_simple_fn(D::zeros(ndim), || unreachable!())
        }
        #[doc = " Create new_array with the right layout for appending to `growing_axis`"]
        #[cold]
        fn change_to_contig_append_layout(&mut self, growing_axis: Axis) {
            let ndim = self.ndim();
            let mut dim = self.raw_dim();
            let mut new_array;
            if growing_axis == Axis(ndim - 1) {
                new_array = Self::uninit(dim.f());
            } else {
                dim.slice_mut()[..=growing_axis.index()].rotate_right(1);
                new_array = Self::uninit(dim);
                new_array.dim.slice_mut()[..=growing_axis.index()].rotate_left(1);
                new_array.strides.slice_mut()[..=growing_axis.index()].rotate_left(1);
            }
            let old_self = std::mem::replace(self, Self::empty());
            old_self.move_into_uninit(new_array.view_mut());
            unsafe {
                *self = new_array.assume_init();
            }
        }
        #[doc = " Append an array to the array along an axis."]
        #[doc = ""]
        #[doc = " The elements of `array` are cloned and extend the axis `axis` in the present array;"]
        #[doc = " `self` will grow in size by `array.len_of(axis)` along `axis`."]
        #[doc = ""]
        #[doc = " ***Errors*** with a shape error if the shape of self does not match the array-to-append;"]
        #[doc = " all axes *except* the axis along which it being appended matter for this check:"]
        #[doc = " the shape of `self` with `axis` removed must be the same as the shape of `array` with"]
        #[doc = " `axis` removed."]
        #[doc = ""]
        #[doc = " The memory layout of the `self` array matters for ensuring that the append is efficient."]
        #[doc = " Appending automatically changes memory layout of the array so that it is appended to"]
        #[doc = " along the \"growing axis\". However, if the memory layout needs adjusting, the array must"]
        #[doc = " reallocate and move memory."]
        #[doc = ""]
        #[doc = " The operation leaves the existing data in place and is most efficent if `axis` is a"]
        #[doc = " \"growing axis\" for the array, i.e. one of these is true:"]
        #[doc = ""]
        #[doc = " - The axis is the longest stride axis, for example the 0th axis in a C-layout or the"]
        #[doc = " *n-1*th axis in an F-layout array."]
        #[doc = " - The axis has length 0 or 1 (It is converted to the new growing axis)"]
        #[doc = ""]
        #[doc = " Ensure appending is efficient by for example starting from an empty array and/or always"]
        #[doc = " appending to an array along the same axis."]
        #[doc = ""]
        #[doc = " The amortized average complexity of the append, when appending along its growing axis, is"]
        #[doc = " O(*m*) where *m* is the number of individual elements to append."]
        #[doc = ""]
        #[doc = " The memory layout of the argument `array` does not matter to the same extent."]
        #[doc = ""]
        #[doc = " ```rust"]
        #[doc = " use ndarray::{Array, ArrayView, array, Axis};"]
        #[doc = ""]
        #[doc = " // create an empty array and append two rows at a time"]
        #[doc = " let mut a = Array::zeros((0, 4));"]
        #[doc = " let ones  = ArrayView::from(&[1.; 8]).into_shape((2, 4)).unwrap();"]
        #[doc = " let zeros = ArrayView::from(&[0.; 8]).into_shape((2, 4)).unwrap();"]
        #[doc = " a.append(Axis(0), ones).unwrap();"]
        #[doc = " a.append(Axis(0), zeros).unwrap();"]
        #[doc = " a.append(Axis(0), ones).unwrap();"]
        #[doc = ""]
        #[doc = " assert_eq!("]
        #[doc = "     a,"]
        #[doc = "     array![[1., 1., 1., 1.],"]
        #[doc = "            [1., 1., 1., 1.],"]
        #[doc = "            [0., 0., 0., 0.],"]
        #[doc = "            [0., 0., 0., 0.],"]
        #[doc = "            [1., 1., 1., 1.],"]
        #[doc = "            [1., 1., 1., 1.]]);"]
        #[doc = " ```"]
        pub fn append(&mut self, axis: Axis, mut array: ArrayView<A, D>) -> Result<(), ShapeError>
        where
            A: Clone,
            D: RemoveAxis,
        {
            if self.ndim() == 0 {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
            let current_axis_len = self.len_of(axis);
            let self_dim = self.raw_dim();
            let array_dim = array.raw_dim();
            let remaining_shape = self_dim.remove_axis(axis);
            let array_rem_shape = array_dim.remove_axis(axis);
            if remaining_shape != array_rem_shape {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
            let len_to_append = array.len();
            let mut res_dim = self_dim;
            res_dim[axis.index()] += array_dim[axis.index()];
            let new_len = dimension::size_of_shape_checked(&res_dim)?;
            if len_to_append == 0 {
                debug_assert_eq!(self.len(), new_len);
                self.dim = res_dim;
                return Ok(());
            }
            let self_is_empty = self.is_empty();
            let mut incompatible_layout = false;
            if !self_is_empty && current_axis_len > 1 {
                let axis_stride = self.stride_of(axis);
                if axis_stride < 0 {
                    incompatible_layout = true;
                } else {
                    for ax in self.axes() {
                        if ax.axis == axis {
                            continue;
                        }
                        if ax.len > 1 && ax.stride.abs() > axis_stride {
                            incompatible_layout = true;
                            break;
                        }
                    }
                }
            }
            if self.len() != self.data.len() {
                incompatible_layout = true;
            }
            if incompatible_layout {
                self.change_to_contig_append_layout(axis);
                debug_assert_eq!(self_is_empty, self.is_empty());
                debug_assert_eq!(current_axis_len, self.len_of(axis));
            }
            let strides = if self_is_empty {
                if axis == Axis(self.ndim() - 1) {
                    res_dim.fortran_strides()
                } else {
                    res_dim.slice_mut()[..=axis.index()].rotate_right(1);
                    let mut strides = res_dim.default_strides();
                    res_dim.slice_mut()[..=axis.index()].rotate_left(1);
                    strides.slice_mut()[..=axis.index()].rotate_left(1);
                    strides
                }
            } else if current_axis_len == 1 {
                let new_stride = self.axes().fold(1, |acc, ax| {
                    if ax.axis == axis || ax.len <= 1 {
                        acc
                    } else {
                        let this_ax = ax.len as isize * ax.stride.abs();
                        if this_ax > acc {
                            this_ax
                        } else {
                            acc
                        }
                    }
                });
                let mut strides = self.strides.clone();
                strides[axis.index()] = new_stride as usize;
                strides
            } else {
                self.strides.clone()
            };
            unsafe {
                let data_to_array_offset = if std::mem::size_of::<A>() != 0 {
                    self.as_ptr().offset_from(self.data.as_ptr())
                } else {
                    0
                };
                debug_assert!(data_to_array_offset >= 0);
                self.ptr = self
                    .data
                    .reserve(len_to_append)
                    .offset(data_to_array_offset);
                let mut tail_strides = strides.clone();
                if tail_strides.ndim() > 1 {
                    for i in 0..tail_strides.ndim() {
                        let s = tail_strides[i] as isize;
                        if s < 0 {
                            tail_strides.set_axis(Axis(i), -s as usize);
                            array.invert_axis(Axis(i));
                        }
                    }
                }
                let tail_ptr = self.data.as_end_nonnull();
                let mut tail_view = RawArrayViewMut::new(tail_ptr, array_dim, tail_strides);
                if tail_view.ndim() > 1 {
                    sort_axes_in_default_order_tandem(&mut tail_view, &mut array);
                    debug_assert!(
                        tail_view.is_standard_layout(),
                        "not std layout dim: {:?}, strides: {:?}",
                        tail_view.shape(),
                        tail_view.strides()
                    );
                }
                struct SetLenOnDrop<'a, A: 'a> {
                    len: usize,
                    data: &'a mut OwnedRepr<A>,
                }
                impl<A> Drop for SetLenOnDrop<'_, A> {
                    fn drop(&mut self) {
                        unsafe {
                            self.data.set_len(self.len);
                        }
                    }
                }
                let mut data_length_guard = SetLenOnDrop {
                    len: self.data.len(),
                    data: &mut self.data,
                };
                Zip::from(tail_view)
                    .and_unchecked(array)
                    .debug_assert_c_order()
                    .for_each(|to, from| {
                        to.write(from.clone());
                        data_length_guard.len += 1;
                    });
                drop(data_length_guard);
                self.strides = strides;
                self.dim = res_dim;
            }
            debug_assert_eq!(self.data.len(), self.len());
            debug_assert_eq!(self.len(), new_len);
            debug_assert!(self.pointer_is_inbounds());
            Ok(())
        }
    }
    #[doc = " This drops all \"unreachable\" elements in `self_` given the data pointer and data length."]
    #[doc = ""]
    #[doc = " # Safety"]
    #[doc = ""]
    #[doc = " This is an internal function for use by move_into and IntoIter only, safety invariants may need"]
    #[doc = " to be upheld across the calls from those implementations."]
    pub(crate) unsafe fn drop_unreachable_raw<A, D>(
        mut self_: RawArrayViewMut<A, D>,
        data_ptr: *mut A,
        data_len: usize,
    ) where
        D: Dimension,
    {
        let self_len = self_.len();
        for i in 0..self_.ndim() {
            if self_.stride_of(Axis(i)) < 0 {
                self_.invert_axis(Axis(i));
            }
        }
        sort_axes_in_default_order(&mut self_);
        let array_memory_head_ptr = self_.ptr.as_ptr();
        let data_end_ptr = data_ptr.add(data_len);
        debug_assert!(data_ptr <= array_memory_head_ptr);
        debug_assert!(array_memory_head_ptr <= data_end_ptr);
        let inner_lane_len;
        if self_.ndim() > 1 && self_.strides.last_elem() == 1 {
            self_.dim.slice_mut().rotate_right(1);
            self_.strides.slice_mut().rotate_right(1);
            inner_lane_len = self_.dim[0];
            self_.dim[0] = 1;
            self_.strides[0] = 1;
        } else {
            inner_lane_len = 1;
        }
        let mut iter = Baseiter::new(self_.ptr.as_ptr(), self_.dim, self_.strides);
        let mut dropped_elements = 0;
        let mut last_ptr = data_ptr;
        while let Some(elem_ptr) = iter.next() {
            while last_ptr != elem_ptr {
                debug_assert!(last_ptr < data_end_ptr);
                std::ptr::drop_in_place(last_ptr);
                last_ptr = last_ptr.add(1);
                dropped_elements += 1;
            }
            last_ptr = elem_ptr.add(inner_lane_len);
        }
        while last_ptr < data_end_ptr {
            std::ptr::drop_in_place(last_ptr);
            last_ptr = last_ptr.add(1);
            dropped_elements += 1;
        }
        assert_eq!(
            data_len,
            dropped_elements + self_len,
            "Internal error: inconsistency in move_into"
        );
    }
    #[doc = " Sort axes to standard order, i.e Axis(0) has biggest stride and Axis(n - 1) least stride"]
    #[doc = ""]
    #[doc = " The axes should have stride >= 0 before calling this method."]
    fn sort_axes_in_default_order<S, D>(a: &mut ArrayBase<S, D>)
    where
        S: RawData,
        D: Dimension,
    {
        if a.ndim() <= 1 {
            return;
        }
        sort_axes1_impl(&mut a.dim, &mut a.strides);
    }
    fn sort_axes1_impl<D>(adim: &mut D, astrides: &mut D)
    where
        D: Dimension,
    {
        debug_assert!(adim.ndim() > 1);
        debug_assert_eq!(adim.ndim(), astrides.ndim());
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..adim.ndim() - 1 {
                let axis_i = i;
                let next_axis = i + 1;
                debug_assert!(astrides.slice()[axis_i] as isize >= 0);
                if (astrides.slice()[axis_i] as isize) < astrides.slice()[next_axis] as isize {
                    changed = true;
                    adim.slice_mut().swap(axis_i, next_axis);
                    astrides.slice_mut().swap(axis_i, next_axis);
                }
            }
        }
    }
    #[doc = " Sort axes to standard order, i.e Axis(0) has biggest stride and Axis(n - 1) least stride"]
    #[doc = ""]
    #[doc = " Axes in a and b are sorted by the strides of `a`, and `a`'s axes should have stride >= 0 before"]
    #[doc = " calling this method."]
    fn sort_axes_in_default_order_tandem<S, S2, D>(
        a: &mut ArrayBase<S, D>,
        b: &mut ArrayBase<S2, D>,
    ) where
        S: RawData,
        S2: RawData,
        D: Dimension,
    {
        if a.ndim() <= 1 {
            return;
        }
        sort_axes2_impl(&mut a.dim, &mut a.strides, &mut b.dim, &mut b.strides);
    }
    fn sort_axes2_impl<D>(adim: &mut D, astrides: &mut D, bdim: &mut D, bstrides: &mut D)
    where
        D: Dimension,
    {
        debug_assert!(adim.ndim() > 1);
        debug_assert_eq!(adim.ndim(), bdim.ndim());
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..adim.ndim() - 1 {
                let axis_i = i;
                let next_axis = i + 1;
                debug_assert!(astrides.slice()[axis_i] as isize >= 0);
                if (astrides.slice()[axis_i] as isize) < astrides.slice()[next_axis] as isize {
                    changed = true;
                    adim.slice_mut().swap(axis_i, next_axis);
                    astrides.slice_mut().swap(axis_i, next_axis);
                    bdim.slice_mut().swap(axis_i, next_axis);
                    bstrides.slice_mut().swap(axis_i, next_axis);
                }
            }
        }
    }
}
mod impl_special_element_types {
    use crate::imp_prelude::*;
    use crate::RawDataSubst;
    use std::mem::MaybeUninit;
    #[doc = " Methods specific to arrays with `MaybeUninit` elements."]
    #[doc = ""]
    #[doc = " ***See also all methods for [`ArrayBase`]***"]
    impl<A, S, D> ArrayBase<S, D>
    where
        S: RawDataSubst<A, Elem = MaybeUninit<A>>,
        D: Dimension,
    {
        #[doc = " **Promise** that the array's elements are all fully initialized, and convert"]
        #[doc = " the array from element type `MaybeUninit<A>` to `A`."]
        #[doc = ""]
        #[doc = " For example, it can convert an `Array<MaybeUninit<f64>, D>` to `Array<f64, D>`."]
        #[doc = ""]
        #[doc = " ## Safety"]
        #[doc = ""]
        #[doc = " Safe to use if all the array's elements have been initialized."]
        #[doc = ""]
        #[doc = " Note that for owned and shared ownership arrays, the promise must include all of the"]
        #[doc = " array's storage; it is for example possible to slice these in place, but that must"]
        #[doc = " only be done after all elements have been initialized."]
        pub unsafe fn assume_init(self) -> ArrayBase<<S as RawDataSubst<A>>::Output, D> {
            let ArrayBase {
                data,
                ptr,
                dim,
                strides,
            } = self;
            let data = S::data_subst(data);
            let ptr = ptr.cast::<A>();
            ArrayBase::from_data_ptr(data, ptr).with_strides_dim(strides, dim)
        }
    }
}
#[doc = " Private Methods"]
impl<A, S, D> ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    #[inline]
    fn broadcast_unwrap<E>(&self, dim: E) -> ArrayView<'_, A, E>
    where
        E: Dimension,
    {
        #[cold]
        #[inline(never)]
        fn broadcast_panic<D, E>(from: &D, to: &E) -> !
        where
            D: Dimension,
            E: Dimension,
        {
            panic!(
                "ndarray: could not broadcast array from shape: {:?} to: {:?}",
                from.slice(),
                to.slice()
            )
        }
        match self.broadcast(dim.clone()) {
            Some(it) => it,
            None => broadcast_panic(&self.dim, &dim),
        }
    }
    #[inline]
    fn broadcast_assume<E>(&self, dim: E) -> ArrayView<'_, A, E>
    where
        E: Dimension,
    {
        let dim = dim.into_dimension();
        debug_assert_eq!(self.shape(), dim.slice());
        let ptr = self.ptr;
        let mut strides = dim.clone();
        strides.slice_mut().copy_from_slice(self.strides.slice());
        unsafe { ArrayView::new(ptr, dim, strides) }
    }
    fn raw_strides(&self) -> D {
        self.strides.clone()
    }
    #[doc = " Remove array axis `axis` and return the result."]
    fn try_remove_axis(self, axis: Axis) -> ArrayBase<S, D::Smaller> {
        let d = self.dim.try_remove_axis(axis);
        let s = self.strides.try_remove_axis(axis);
        unsafe { self.with_strides_dim(s, d) }
    }
}
mod impl_1d {
    #![doc = " Methods for one-dimensional arrays."]
    use crate::imp_prelude::*;
    use crate::low_level_util::AbortIfPanic;
    use alloc::vec::Vec;
    use std::mem::MaybeUninit;
    #[doc = " # Methods For 1-D Arrays"]
    impl<A, S> ArrayBase<S, Ix1> where S: RawData<Elem = A> {}
}
mod impl_ops {
    use crate::dimension::DimMax;
    use crate::Zip;
    use num_complex::Complex;
    #[doc = " Elements that can be used as direct operands in arithmetic with arrays."]
    #[doc = ""]
    #[doc = " For example, `f64` is a `ScalarOperand` which means that for an array `a`,"]
    #[doc = " arithmetic like `a + 1.0`, and, `a * 2.`, and `a += 3.` are allowed."]
    #[doc = ""]
    #[doc = " In the description below, let `A` be an array or array view,"]
    #[doc = " let `B` be an array with owned data,"]
    #[doc = " and let `C` be an array with mutable data."]
    #[doc = ""]
    #[doc = " `ScalarOperand` determines for which scalars `K` operations `&A @ K`, and `B @ K`,"]
    #[doc = " and `C @= K` are defined, as ***right hand side operands***, for applicable"]
    #[doc = " arithmetic operators (denoted `@`)."]
    #[doc = ""]
    #[doc = " ***Left hand side*** scalar operands are not related to this trait"]
    #[doc = " (they need one `impl` per concrete scalar type); but they are still"]
    #[doc = " implemented for the same types, allowing operations"]
    #[doc = " `K @ &A`, and `K @ B` for primitive numeric types `K`."]
    #[doc = ""]
    #[doc = " This trait ***does not*** limit which elements can be stored in an array in general."]
    #[doc = " Non-`ScalarOperand` types can still participate in arithmetic as array elements in"]
    #[doc = " in array-array operations."]
    pub trait ScalarOperand: 'static + Clone {}
    impl ScalarOperand for bool {}
    impl ScalarOperand for i8 {}
    impl ScalarOperand for u8 {}
    impl ScalarOperand for i16 {}
    impl ScalarOperand for u16 {}
    impl ScalarOperand for i32 {}
    impl ScalarOperand for u32 {}
    impl ScalarOperand for i64 {}
    impl ScalarOperand for u64 {}
    impl ScalarOperand for i128 {}
    impl ScalarOperand for u128 {}
    impl ScalarOperand for isize {}
    impl ScalarOperand for usize {}
    impl ScalarOperand for f32 {}
    impl ScalarOperand for f64 {}
    impl ScalarOperand for Complex<f32> {}
    impl ScalarOperand for Complex<f64> {}
    macro_rules ! impl_binary_op (($ trt : ident , $ operator : tt , $ mth : ident , $ iop : tt , $ doc : expr) => (# [doc = " Perform elementwise"] # [doc =$ doc] # [doc = " between `self` and `rhs`,"] # [doc = " and return the result."] # [doc = ""] # [doc = " `self` must be an `Array` or `ArcArray`."] # [doc = ""] # [doc = " If their shapes disagree, `self` is broadcast to their broadcast shape."] # [doc = ""] # [doc = " **Panics** if broadcasting isn’t possible."] impl < A , B , S , S2 , D , E > $ trt < ArrayBase < S2 , E >> for ArrayBase < S , D > where A : Clone + $ trt < B , Output = A >, B : Clone , S : DataOwned < Elem = A > + DataMut , S2 : Data < Elem = B >, D : Dimension + DimMax < E >, E : Dimension , { type Output = ArrayBase < S , < D as DimMax < E >>:: Output >; fn $ mth (self , rhs : ArrayBase < S2 , E >) -> Self :: Output { self .$ mth (& rhs) } } # [doc = " Perform elementwise"] # [doc =$ doc] # [doc = " between `self` and reference `rhs`,"] # [doc = " and return the result."] # [doc = ""] # [doc = " `rhs` must be an `Array` or `ArcArray`."] # [doc = ""] # [doc = " If their shapes disagree, `self` is broadcast to their broadcast shape,"] # [doc = " cloning the data if needed."] # [doc = ""] # [doc = " **Panics** if broadcasting isn’t possible."] impl <'a , A , B , S , S2 , D , E > $ trt <&'a ArrayBase < S2 , E >> for ArrayBase < S , D > where A : Clone + $ trt < B , Output = A >, B : Clone , S : DataOwned < Elem = A > + DataMut , S2 : Data < Elem = B >, D : Dimension + DimMax < E >, E : Dimension , { type Output = ArrayBase < S , < D as DimMax < E >>:: Output >; fn $ mth (self , rhs : & ArrayBase < S2 , E >) -> Self :: Output { if self . ndim () == rhs . ndim () && self . shape () == rhs . shape () { let mut out = self . into_dimensionality ::<< D as DimMax < E >>:: Output > () . unwrap () ; out . zip_mut_with_same_shape (rhs , clone_iopf (A ::$ mth)) ; out } else { let (lhs_view , rhs_view) = self . broadcast_with (& rhs) . unwrap () ; if lhs_view . shape () == self . shape () { let mut out = self . into_dimensionality ::<< D as DimMax < E >>:: Output > () . unwrap () ; out . zip_mut_with_same_shape (& rhs_view , clone_iopf (A ::$ mth)) ; out } else { Zip :: from (& lhs_view) . and (& rhs_view) . map_collect_owned (clone_opf (A ::$ mth)) } } } } # [doc = " Perform elementwise"] # [doc =$ doc] # [doc = " between reference `self` and `rhs`,"] # [doc = " and return the result."] # [doc = ""] # [doc = " `rhs` must be an `Array` or `ArcArray`."] # [doc = ""] # [doc = " If their shapes disagree, `self` is broadcast to their broadcast shape,"] # [doc = " cloning the data if needed."] # [doc = ""] # [doc = " **Panics** if broadcasting isn’t possible."] impl <'a , A , B , S , S2 , D , E > $ trt < ArrayBase < S2 , E >> for &'a ArrayBase < S , D > where A : Clone + $ trt < B , Output = B >, B : Clone , S : Data < Elem = A >, S2 : DataOwned < Elem = B > + DataMut , D : Dimension , E : Dimension + DimMax < D >, { type Output = ArrayBase < S2 , < E as DimMax < D >>:: Output >; fn $ mth (self , rhs : ArrayBase < S2 , E >) -> Self :: Output where { if self . ndim () == rhs . ndim () && self . shape () == rhs . shape () { let mut out = rhs . into_dimensionality ::<< E as DimMax < D >>:: Output > () . unwrap () ; out . zip_mut_with_same_shape (self , clone_iopf_rev (A ::$ mth)) ; out } else { let (rhs_view , lhs_view) = rhs . broadcast_with (self) . unwrap () ; if rhs_view . shape () == rhs . shape () { let mut out = rhs . into_dimensionality ::<< E as DimMax < D >>:: Output > () . unwrap () ; out . zip_mut_with_same_shape (& lhs_view , clone_iopf_rev (A ::$ mth)) ; out } else { Zip :: from (& lhs_view) . and (& rhs_view) . map_collect_owned (clone_opf (A ::$ mth)) } } } } # [doc = " Perform elementwise"] # [doc =$ doc] # [doc = " between references `self` and `rhs`,"] # [doc = " and return the result as a new `Array`."] # [doc = ""] # [doc = " If their shapes disagree, `self` and `rhs` is broadcast to their broadcast shape,"] # [doc = " cloning the data if needed."] # [doc = ""] # [doc = " **Panics** if broadcasting isn’t possible."] impl <'a , A , B , S , S2 , D , E > $ trt <&'a ArrayBase < S2 , E >> for &'a ArrayBase < S , D > where A : Clone + $ trt < B , Output = A >, B : Clone , S : Data < Elem = A >, S2 : Data < Elem = B >, D : Dimension + DimMax < E >, E : Dimension , { type Output = Array < A , < D as DimMax < E >>:: Output >; fn $ mth (self , rhs : &'a ArrayBase < S2 , E >) -> Self :: Output { let (lhs , rhs) = if self . ndim () == rhs . ndim () && self . shape () == rhs . shape () { let lhs = self . view () . into_dimensionality ::<< D as DimMax < E >>:: Output > () . unwrap () ; let rhs = rhs . view () . into_dimensionality ::<< D as DimMax < E >>:: Output > () . unwrap () ; (lhs , rhs) } else { self . broadcast_with (rhs) . unwrap () } ; Zip :: from (lhs) . and (rhs) . map_collect (clone_opf (A ::$ mth)) } } # [doc = " Perform elementwise"] # [doc =$ doc] # [doc = " between `self` and the scalar `x`,"] # [doc = " and return the result (based on `self`)."] # [doc = ""] # [doc = " `self` must be an `Array` or `ArcArray`."] impl < A , S , D , B > $ trt < B > for ArrayBase < S , D > where A : Clone + $ trt < B , Output = A >, S : DataOwned < Elem = A > + DataMut , D : Dimension , B : ScalarOperand , { type Output = ArrayBase < S , D >; fn $ mth (mut self , x : B) -> ArrayBase < S , D > { self . map_inplace (move | elt | { * elt = elt . clone () $ operator x . clone () ; }) ; self } } # [doc = " Perform elementwise"] # [doc =$ doc] # [doc = " between the reference `self` and the scalar `x`,"] # [doc = " and return the result as a new `Array`."] impl <'a , A , S , D , B > $ trt < B > for &'a ArrayBase < S , D > where A : Clone + $ trt < B , Output = A >, S : Data < Elem = A >, D : Dimension , B : ScalarOperand , { type Output = Array < A , D >; fn $ mth (self , x : B) -> Self :: Output { self . map (move | elt | elt . clone () $ operator x . clone ()) } }) ;) ;
    macro_rules! if_commutative {
        (Commute { $ a : expr } or { $ b : expr }) => {
            $a
        };
        (Ordered { $ a : expr } or { $ b : expr }) => {
            $b
        };
    }
    macro_rules ! impl_scalar_lhs_op { ($ scalar : ty , $ commutative : ident , $ operator : tt , $ trt : ident , $ mth : ident , $ doc : expr) => (impl < S , D > $ trt < ArrayBase < S , D >> for $ scalar where S : DataOwned < Elem =$ scalar > + DataMut , D : Dimension , { type Output = ArrayBase < S , D >; fn $ mth (self , rhs : ArrayBase < S , D >) -> ArrayBase < S , D > { if_commutative ! ($ commutative { rhs .$ mth (self) } or { { let mut rhs = rhs ; rhs . map_inplace (move | elt | { * elt = self $ operator * elt ; }) ; rhs } }) } } impl <'a , S , D > $ trt <&'a ArrayBase < S , D >> for $ scalar where S : Data < Elem =$ scalar >, D : Dimension , { type Output = Array <$ scalar , D >; fn $ mth (self , rhs : & ArrayBase < S , D >) -> Self :: Output { if_commutative ! ($ commutative { rhs .$ mth (self) } or { rhs . map (move | elt | self . clone () $ operator elt . clone ()) }) } }) ; }
}
pub use crate::impl_ops::ScalarOperand;
mod impl_views {
    mod constructors {
        use crate::dimension;
        use crate::dimension::offset_from_low_addr_ptr_to_logical_ptr;
        use crate::error::ShapeError;
        use crate::extension::nonnull::nonnull_debug_checked_from_ptr;
        use crate::imp_prelude::*;
        use crate::{is_aligned, StrideShape};
        use std::ptr::NonNull;
        #[doc = " Methods for read-only array views."]
        impl<'a, A, D> ArrayView<'a, A, D>
        where
            D: Dimension,
        {
            #[doc = " Create an `ArrayView<A, D>` from shape information and a raw pointer to"]
            #[doc = " the elements."]
            #[doc = ""]
            #[doc = " # Safety"]
            #[doc = ""]
            #[doc = " The caller is responsible for ensuring all of the following:"]
            #[doc = ""]
            #[doc = " * The elements seen by moving `ptr` according to the shape and strides"]
            #[doc = "   must live at least as long as `'a` and must not be not mutably"]
            #[doc = "   aliased for the duration of `'a`."]
            #[doc = ""]
            #[doc = " * `ptr` must be non-null and aligned, and it must be safe to"]
            #[doc = "   [`.offset()`] `ptr` by zero."]
            #[doc = ""]
            #[doc = " * It must be safe to [`.offset()`] the pointer repeatedly along all"]
            #[doc = "   axes and calculate the `count`s for the `.offset()` calls without"]
            #[doc = "   overflow, even if the array is empty or the elements are zero-sized."]
            #[doc = ""]
            #[doc = "   In other words,"]
            #[doc = ""]
            #[doc = "   * All possible pointers generated by moving along all axes must be in"]
            #[doc = "     bounds or one byte past the end of a single allocation with element"]
            #[doc = "     type `A`. The only exceptions are if the array is empty or the element"]
            #[doc = "     type is zero-sized. In these cases, `ptr` may be dangling, but it must"]
            #[doc = "     still be safe to [`.offset()`] the pointer along the axes."]
            #[doc = ""]
            #[doc = "   * The offset in units of bytes between the least address and greatest"]
            #[doc = "     address by moving along all axes must not exceed `isize::MAX`. This"]
            #[doc = "     constraint prevents the computed offset, in bytes, from overflowing"]
            #[doc = "     `isize` regardless of the starting point due to past offsets."]
            #[doc = ""]
            #[doc = "   * The offset in units of `A` between the least address and greatest"]
            #[doc = "     address by moving along all axes must not exceed `isize::MAX`. This"]
            #[doc = "     constraint prevents overflow when calculating the `count` parameter to"]
            #[doc = "     [`.offset()`] regardless of the starting point due to past offsets."]
            #[doc = ""]
            #[doc = " * The product of non-zero axis lengths must not exceed `isize::MAX`."]
            #[doc = ""]
            #[doc = " * Strides must be non-negative."]
            #[doc = ""]
            #[doc = " This function can use debug assertions to check some of these requirements,"]
            #[doc = " but it's not a complete check."]
            #[doc = ""]
            #[doc = " [`.offset()`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset"]
            pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *const A) -> Self
            where
                Sh: Into<StrideShape<D>>,
            {
                RawArrayView::from_shape_ptr(shape, ptr).deref_into_view()
            }
        }
        #[doc = " Methods for read-write array views."]
        impl<'a, A, D> ArrayViewMut<'a, A, D>
        where
            D: Dimension,
        {
            #[doc = " Create an `ArrayViewMut<A, D>` from shape information and a"]
            #[doc = " raw pointer to the elements."]
            #[doc = ""]
            #[doc = " # Safety"]
            #[doc = ""]
            #[doc = " The caller is responsible for ensuring all of the following:"]
            #[doc = ""]
            #[doc = " * The elements seen by moving `ptr` according to the shape and strides"]
            #[doc = "   must live at least as long as `'a` and must not be aliased for the"]
            #[doc = "   duration of `'a`."]
            #[doc = ""]
            #[doc = " * `ptr` must be non-null and aligned, and it must be safe to"]
            #[doc = "   [`.offset()`] `ptr` by zero."]
            #[doc = ""]
            #[doc = " * It must be safe to [`.offset()`] the pointer repeatedly along all"]
            #[doc = "   axes and calculate the `count`s for the `.offset()` calls without"]
            #[doc = "   overflow, even if the array is empty or the elements are zero-sized."]
            #[doc = ""]
            #[doc = "   In other words,"]
            #[doc = ""]
            #[doc = "   * All possible pointers generated by moving along all axes must be in"]
            #[doc = "     bounds or one byte past the end of a single allocation with element"]
            #[doc = "     type `A`. The only exceptions are if the array is empty or the element"]
            #[doc = "     type is zero-sized. In these cases, `ptr` may be dangling, but it must"]
            #[doc = "     still be safe to [`.offset()`] the pointer along the axes."]
            #[doc = ""]
            #[doc = "   * The offset in units of bytes between the least address and greatest"]
            #[doc = "     address by moving along all axes must not exceed `isize::MAX`. This"]
            #[doc = "     constraint prevents the computed offset, in bytes, from overflowing"]
            #[doc = "     `isize` regardless of the starting point due to past offsets."]
            #[doc = ""]
            #[doc = "   * The offset in units of `A` between the least address and greatest"]
            #[doc = "     address by moving along all axes must not exceed `isize::MAX`. This"]
            #[doc = "     constraint prevents overflow when calculating the `count` parameter to"]
            #[doc = "     [`.offset()`] regardless of the starting point due to past offsets."]
            #[doc = ""]
            #[doc = " * The product of non-zero axis lengths must not exceed `isize::MAX`."]
            #[doc = ""]
            #[doc = " * Strides must be non-negative."]
            #[doc = ""]
            #[doc = " This function can use debug assertions to check some of these requirements,"]
            #[doc = " but it's not a complete check."]
            #[doc = ""]
            #[doc = " [`.offset()`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset"]
            pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *mut A) -> Self
            where
                Sh: Into<StrideShape<D>>,
            {
                RawArrayViewMut::from_shape_ptr(shape, ptr).deref_into_view_mut()
            }
        }
        #[doc = " Private array view methods"]
        impl<'a, A, D> ArrayView<'a, A, D>
        where
            D: Dimension,
        {
            #[doc = " Create a new `ArrayView`"]
            #[doc = ""]
            #[doc = " Unsafe because: `ptr` must be valid for the given dimension and strides."]
            #[inline(always)]
            pub(crate) unsafe fn new(ptr: NonNull<A>, dim: D, strides: D) -> Self {
                if cfg!(debug_assertions) {
                    assert!(is_aligned(ptr.as_ptr()), "The pointer must be aligned.");
                    dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides).unwrap();
                }
                ArrayView::from_data_ptr(ViewRepr::new(), ptr).with_strides_dim(strides, dim)
            }
            #[doc = " Unsafe because: `ptr` must be valid for the given dimension and strides."]
            #[inline]
            pub(crate) unsafe fn new_(ptr: *const A, dim: D, strides: D) -> Self {
                Self::new(nonnull_debug_checked_from_ptr(ptr as *mut A), dim, strides)
            }
        }
        impl<'a, A, D> ArrayViewMut<'a, A, D>
        where
            D: Dimension,
        {
            #[doc = " Create a new `ArrayView`"]
            #[doc = ""]
            #[doc = " Unsafe because: `ptr` must be valid for the given dimension and strides."]
            #[inline(always)]
            pub(crate) unsafe fn new(ptr: NonNull<A>, dim: D, strides: D) -> Self {
                if cfg!(debug_assertions) {
                    assert!(is_aligned(ptr.as_ptr()), "The pointer must be aligned.");
                    dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides).unwrap();
                }
                ArrayViewMut::from_data_ptr(ViewRepr::new(), ptr).with_strides_dim(strides, dim)
            }
            #[doc = " Create a new `ArrayView`"]
            #[doc = ""]
            #[doc = " Unsafe because: `ptr` must be valid for the given dimension and strides."]
            #[inline(always)]
            pub(crate) unsafe fn new_(ptr: *mut A, dim: D, strides: D) -> Self {
                Self::new(nonnull_debug_checked_from_ptr(ptr), dim, strides)
            }
        }
    }
    mod conversions {
        use crate::dimension::offset_from_low_addr_ptr_to_logical_ptr;
        use crate::imp_prelude::*;
        use crate::iter::{self, AxisIter, AxisIterMut};
        use crate::math_cell::MathCell;
        use crate::IndexLonger;
        use crate::{Baseiter, ElementsBase, ElementsBaseMut, Iter, IterMut};
        use alloc::slice;
        use rawpointer::PointerExt;
        use std::mem::MaybeUninit;
        #[doc = " Methods for read-only array views."]
        impl<'a, A, D> ArrayView<'a, A, D>
        where
            D: Dimension,
        {
            #[doc = " Return the array’s data as a slice, if it is contiguous and in standard order."]
            #[doc = " Return `None` otherwise."]
            #[doc = ""]
            #[doc = " Note that while the method is similar to [`ArrayBase::as_slice()`], this method transfers"]
            #[doc = " the view's lifetime to the slice, so it is a bit more powerful."]
            pub fn to_slice(&self) -> Option<&'a [A]> {
                if self.is_standard_layout() {
                    unsafe { Some(slice::from_raw_parts(self.ptr.as_ptr(), self.len())) }
                } else {
                    None
                }
            }
        }
        #[doc = " Private array view methods"]
        impl<'a, A, D> ArrayView<'a, A, D>
        where
            D: Dimension,
        {
            #[inline]
            pub(crate) fn into_base_iter(self) -> Baseiter<A, D> {
                unsafe { Baseiter::new(self.ptr.as_ptr(), self.dim, self.strides) }
            }
            #[inline]
            pub(crate) fn into_elements_base(self) -> ElementsBase<'a, A, D> {
                ElementsBase::new(self)
            }
            pub(crate) fn into_iter_(self) -> Iter<'a, A, D> {
                Iter::new(self)
            }
        }
        impl<'a, A, D> ArrayViewMut<'a, A, D>
        where
            D: Dimension,
        {
            pub(crate) fn into_view(self) -> ArrayView<'a, A, D> {
                unsafe { ArrayView::new(self.ptr, self.dim, self.strides) }
            }
            #[doc = " Converts to a mutable raw array view."]
            pub(crate) fn into_raw_view_mut(self) -> RawArrayViewMut<A, D> {
                unsafe { RawArrayViewMut::new(self.ptr, self.dim, self.strides) }
            }
            #[inline]
            pub(crate) fn into_base_iter(self) -> Baseiter<A, D> {
                unsafe { Baseiter::new(self.ptr.as_ptr(), self.dim, self.strides) }
            }
            #[inline]
            pub(crate) fn into_elements_base(self) -> ElementsBaseMut<'a, A, D> {
                ElementsBaseMut::new(self)
            }
            #[doc = " Return the array’s data as a slice, if it is contiguous and in standard order."]
            #[doc = " Otherwise return self in the Err branch of the result."]
            pub(crate) fn try_into_slice(self) -> Result<&'a mut [A], Self> {
                if self.is_standard_layout() {
                    unsafe { Ok(slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len())) }
                } else {
                    Err(self)
                }
            }
            pub(crate) fn into_iter_(self) -> IterMut<'a, A, D> {
                IterMut::new(self)
            }
        }
    }
    mod indexing {
        use crate::arraytraits::array_out_of_bounds;
        use crate::imp_prelude::*;
        use crate::NdIndex;
        #[doc = " Extra indexing methods for array views"]
        #[doc = ""]
        #[doc = " These methods are very similar to regular indexing or calling of the"]
        #[doc = " `get`/`get_mut` methods that we can use on any array or array view. The"]
        #[doc = " difference here is in the length of lifetime in the resulting reference."]
        #[doc = ""]
        #[doc = " **Note** that the `ArrayView` (read-only) and `ArrayViewMut` (read-write) differ"]
        #[doc = " in how they are allowed implement this trait -- `ArrayView`'s implementation"]
        #[doc = " is usual. If you put in a `ArrayView<'a, T, D>` here, you get references"]
        #[doc = " `&'a T` out."]
        #[doc = ""]
        #[doc = " For `ArrayViewMut` to obey the borrowing rules we have to consume the"]
        #[doc = " view if we call any of these methods. (The equivalent of reborrow is"]
        #[doc = " `.view_mut()` for read-write array views, but if you can use that,"]
        #[doc = " then the regular indexing / `get_mut` should suffice, too.)"]
        #[doc = ""]
        #[doc = " ```"]
        #[doc = " use ndarray::IndexLonger;"]
        #[doc = " use ndarray::ArrayView;"]
        #[doc = ""]
        #[doc = " let data = [0.; 256];"]
        #[doc = " let long_life_ref = {"]
        #[doc = "     // make a 16 × 16 array view"]
        #[doc = "     let view = ArrayView::from(&data[..]).into_shape((16, 16)).unwrap();"]
        #[doc = ""]
        #[doc = "     // index the view and with `IndexLonger`."]
        #[doc = "     // Note here that we get a reference with a life that is derived from"]
        #[doc = "     // `data`, the base data, instead of being derived from the view"]
        #[doc = "     IndexLonger::index(&view, [0, 1])"]
        #[doc = " };"]
        #[doc = ""]
        #[doc = " // view goes out of scope"]
        #[doc = ""]
        #[doc = " assert_eq!(long_life_ref, &0.);"]
        #[doc = ""]
        #[doc = " ```"]
        pub trait IndexLonger<I> {
            #[doc = " The type of the reference to the element that is produced, including"]
            #[doc = " its lifetime."]
            type Output;
            #[doc = " Get a reference of a element through the view."]
            #[doc = ""]
            #[doc = " This method is like `Index::index` but with a longer lifetime (matching"]
            #[doc = " the array view); which we can only do for the array view and not in the"]
            #[doc = " `Index` trait."]
            #[doc = ""]
            #[doc = " See also [the `get` method][1] which works for all arrays and array"]
            #[doc = " views."]
            #[doc = ""]
            #[doc = " [1]: ArrayBase::get"]
            #[doc = ""]
            #[doc = " **Panics** if index is out of bounds."]
            fn index(self, index: I) -> Self::Output;
            #[doc = " Get a reference of a element through the view."]
            #[doc = ""]
            #[doc = " This method is like `ArrayBase::get` but with a longer lifetime (matching"]
            #[doc = " the array view); which we can only do for the array view and not in the"]
            #[doc = " `Index` trait."]
            #[doc = ""]
            #[doc = " See also [the `get` method][1] (and [`get_mut`][2]) which works for all arrays and array"]
            #[doc = " views."]
            #[doc = ""]
            #[doc = " [1]: ArrayBase::get"]
            #[doc = " [2]: ArrayBase::get_mut"]
            #[doc = ""]
            #[doc = " **Panics** if index is out of bounds."]
            fn get(self, index: I) -> Option<Self::Output>;
            #[doc = " Get a reference of a element through the view without boundary check"]
            #[doc = ""]
            #[doc = " This method is like `elem` with a longer lifetime (matching the array"]
            #[doc = " view); which we can't do for general arrays."]
            #[doc = ""]
            #[doc = " See also [the `uget` method][1] which works for all arrays and array"]
            #[doc = " views."]
            #[doc = ""]
            #[doc = " [1]: ArrayBase::uget"]
            #[doc = ""]
            #[doc = " **Note:** only unchecked for non-debug builds of ndarray."]
            #[doc = ""]
            #[doc = " # Safety"]
            #[doc = ""]
            #[doc = " The caller must ensure that the index is in-bounds."]
            unsafe fn uget(self, index: I) -> Self::Output;
        }
    }
    mod splitting {
        use crate::imp_prelude::*;
        use crate::slice::MultiSliceArg;
        use num_complex::Complex;
    }
    pub use indexing::*;
}
mod impl_raw_views {
    use crate::dimension::{self, stride_offset};
    use crate::extension::nonnull::nonnull_debug_checked_from_ptr;
    use crate::imp_prelude::*;
    use crate::is_aligned;
    use crate::shape_builder::{StrideShape, Strides};
    use num_complex::Complex;
    use std::mem;
    use std::ptr::NonNull;
    impl<A, D> RawArrayView<A, D>
    where
        D: Dimension,
    {
        #[doc = " Create a new `RawArrayView`."]
        #[doc = ""]
        #[doc = " Unsafe because caller is responsible for ensuring that the array will"]
        #[doc = " meet all of the invariants of the `ArrayBase` type."]
        #[inline]
        pub(crate) unsafe fn new(ptr: NonNull<A>, dim: D, strides: D) -> Self {
            RawArrayView::from_data_ptr(RawViewRepr::new(), ptr).with_strides_dim(strides, dim)
        }
        unsafe fn new_(ptr: *const A, dim: D, strides: D) -> Self {
            Self::new(nonnull_debug_checked_from_ptr(ptr as *mut A), dim, strides)
        }
        #[doc = " Create an `RawArrayView<A, D>` from shape information and a raw pointer"]
        #[doc = " to the elements."]
        #[doc = ""]
        #[doc = " # Safety"]
        #[doc = ""]
        #[doc = " The caller is responsible for ensuring all of the following:"]
        #[doc = ""]
        #[doc = " * `ptr` must be non-null, and it must be safe to [`.offset()`] `ptr` by"]
        #[doc = "   zero."]
        #[doc = ""]
        #[doc = " * It must be safe to [`.offset()`] the pointer repeatedly along all"]
        #[doc = "   axes and calculate the `count`s for the `.offset()` calls without"]
        #[doc = "   overflow, even if the array is empty or the elements are zero-sized."]
        #[doc = ""]
        #[doc = "   In other words,"]
        #[doc = ""]
        #[doc = "   * All possible pointers generated by moving along all axes must be in"]
        #[doc = "     bounds or one byte past the end of a single allocation with element"]
        #[doc = "     type `A`. The only exceptions are if the array is empty or the element"]
        #[doc = "     type is zero-sized. In these cases, `ptr` may be dangling, but it must"]
        #[doc = "     still be safe to [`.offset()`] the pointer along the axes."]
        #[doc = ""]
        #[doc = "   * The offset in units of bytes between the least address and greatest"]
        #[doc = "     address by moving along all axes must not exceed `isize::MAX`. This"]
        #[doc = "     constraint prevents the computed offset, in bytes, from overflowing"]
        #[doc = "     `isize` regardless of the starting point due to past offsets."]
        #[doc = ""]
        #[doc = "   * The offset in units of `A` between the least address and greatest"]
        #[doc = "     address by moving along all axes must not exceed `isize::MAX`. This"]
        #[doc = "     constraint prevents overflow when calculating the `count` parameter to"]
        #[doc = "     [`.offset()`] regardless of the starting point due to past offsets."]
        #[doc = ""]
        #[doc = " * The product of non-zero axis lengths must not exceed `isize::MAX`."]
        #[doc = " "]
        #[doc = " * Strides must be non-negative."]
        #[doc = ""]
        #[doc = " This function can use debug assertions to check some of these requirements,"]
        #[doc = " but it's not a complete check."]
        #[doc = ""]
        #[doc = " [`.offset()`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset"]
        pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *const A) -> Self
        where
            Sh: Into<StrideShape<D>>,
        {
            let shape = shape.into();
            let dim = shape.dim;
            if cfg!(debug_assertions) {
                assert!(!ptr.is_null(), "The pointer must be non-null.");
                if let Strides::Custom(strides) = &shape.strides {
                    dimension::strides_non_negative(strides).unwrap();
                    dimension::max_abs_offset_check_overflow::<A, _>(&dim, strides).unwrap();
                } else {
                    dimension::size_of_shape_checked(&dim).unwrap();
                }
            }
            let strides = shape.strides.strides_for_dim(&dim);
            RawArrayView::new_(ptr, dim, strides)
        }
        #[doc = " Converts to a read-only view of the array."]
        #[doc = ""]
        #[doc = " # Safety"]
        #[doc = ""]
        #[doc = " From a safety standpoint, this is equivalent to dereferencing a raw"]
        #[doc = " pointer for every element in the array. You must ensure that all of the"]
        #[doc = " data is valid, ensure that the pointer is aligned, and choose the"]
        #[doc = " correct lifetime."]
        #[inline]
        pub unsafe fn deref_into_view<'a>(self) -> ArrayView<'a, A, D> {
            debug_assert!(
                is_aligned(self.ptr.as_ptr()),
                "The pointer must be aligned."
            );
            ArrayView::new(self.ptr, self.dim, self.strides)
        }
        #[doc = " Split the array view along `axis` and return one array pointer strictly"]
        #[doc = " before the split and one array pointer after the split."]
        #[doc = ""]
        #[doc = " **Panics** if `axis` or `index` is out of bounds."]
        pub fn split_at(self, axis: Axis, index: Ix) -> (Self, Self) {
            assert!(index <= self.len_of(axis));
            let left_ptr = self.ptr.as_ptr();
            let right_ptr = if index == self.len_of(axis) {
                self.ptr.as_ptr()
            } else {
                let offset = stride_offset(index, self.strides.axis(axis));
                unsafe { self.ptr.as_ptr().offset(offset) }
            };
            let mut dim_left = self.dim.clone();
            dim_left.set_axis(axis, index);
            let left = unsafe { Self::new_(left_ptr, dim_left, self.strides.clone()) };
            let mut dim_right = self.dim;
            let right_len = dim_right.axis(axis) - index;
            dim_right.set_axis(axis, right_len);
            let right = unsafe { Self::new_(right_ptr, dim_right, self.strides) };
            (left, right)
        }
    }
    impl<T, D> RawArrayView<Complex<T>, D>
    where
        D: Dimension,
    {
        #[doc = " Splits the view into views of the real and imaginary components of the"]
        #[doc = " elements."]
        pub fn split_complex(self) -> Complex<RawArrayView<T, D>> {
            assert_eq!(
                mem::size_of::<Complex<T>>(),
                mem::size_of::<T>().checked_mul(2).unwrap()
            );
            assert_eq!(mem::align_of::<Complex<T>>(), mem::align_of::<T>());
            let dim = self.dim.clone();
            let mut strides = self.strides.clone();
            if mem::size_of::<T>() != 0 {
                for ax in 0..strides.ndim() {
                    if dim[ax] > 1 {
                        strides[ax] = (strides[ax] as isize * 2) as usize;
                    }
                }
            }
            let ptr_re: *mut T = self.ptr.as_ptr().cast();
            let ptr_im: *mut T = if self.is_empty() {
                ptr_re
            } else {
                unsafe { ptr_re.add(1) }
            };
            unsafe {
                Complex {
                    re: RawArrayView::new_(ptr_re, dim.clone(), strides.clone()),
                    im: RawArrayView::new_(ptr_im, dim, strides),
                }
            }
        }
    }
    impl<A, D> RawArrayViewMut<A, D>
    where
        D: Dimension,
    {
        #[doc = " Create a new `RawArrayViewMut`."]
        #[doc = ""]
        #[doc = " Unsafe because caller is responsible for ensuring that the array will"]
        #[doc = " meet all of the invariants of the `ArrayBase` type."]
        #[inline]
        pub(crate) unsafe fn new(ptr: NonNull<A>, dim: D, strides: D) -> Self {
            RawArrayViewMut::from_data_ptr(RawViewRepr::new(), ptr).with_strides_dim(strides, dim)
        }
        unsafe fn new_(ptr: *mut A, dim: D, strides: D) -> Self {
            Self::new(nonnull_debug_checked_from_ptr(ptr), dim, strides)
        }
        #[doc = " Create an `RawArrayViewMut<A, D>` from shape information and a raw"]
        #[doc = " pointer to the elements."]
        #[doc = ""]
        #[doc = " # Safety"]
        #[doc = ""]
        #[doc = " The caller is responsible for ensuring all of the following:"]
        #[doc = ""]
        #[doc = " * `ptr` must be non-null, and it must be safe to [`.offset()`] `ptr` by"]
        #[doc = "   zero."]
        #[doc = ""]
        #[doc = " * It must be safe to [`.offset()`] the pointer repeatedly along all"]
        #[doc = "   axes and calculate the `count`s for the `.offset()` calls without"]
        #[doc = "   overflow, even if the array is empty or the elements are zero-sized."]
        #[doc = ""]
        #[doc = "   In other words,"]
        #[doc = ""]
        #[doc = "   * All possible pointers generated by moving along all axes must be in"]
        #[doc = "     bounds or one byte past the end of a single allocation with element"]
        #[doc = "     type `A`. The only exceptions are if the array is empty or the element"]
        #[doc = "     type is zero-sized. In these cases, `ptr` may be dangling, but it must"]
        #[doc = "     still be safe to [`.offset()`] the pointer along the axes."]
        #[doc = ""]
        #[doc = "   * The offset in units of bytes between the least address and greatest"]
        #[doc = "     address by moving along all axes must not exceed `isize::MAX`. This"]
        #[doc = "     constraint prevents the computed offset, in bytes, from overflowing"]
        #[doc = "     `isize` regardless of the starting point due to past offsets."]
        #[doc = ""]
        #[doc = "   * The offset in units of `A` between the least address and greatest"]
        #[doc = "     address by moving along all axes must not exceed `isize::MAX`. This"]
        #[doc = "     constraint prevents overflow when calculating the `count` parameter to"]
        #[doc = "     [`.offset()`] regardless of the starting point due to past offsets."]
        #[doc = ""]
        #[doc = " * The product of non-zero axis lengths must not exceed `isize::MAX`."]
        #[doc = " "]
        #[doc = " * Strides must be non-negative."]
        #[doc = ""]
        #[doc = " This function can use debug assertions to check some of these requirements,"]
        #[doc = " but it's not a complete check."]
        #[doc = ""]
        #[doc = " [`.offset()`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset"]
        pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *mut A) -> Self
        where
            Sh: Into<StrideShape<D>>,
        {
            let shape = shape.into();
            let dim = shape.dim;
            if cfg!(debug_assertions) {
                assert!(!ptr.is_null(), "The pointer must be non-null.");
                if let Strides::Custom(strides) = &shape.strides {
                    dimension::strides_non_negative(strides).unwrap();
                    dimension::max_abs_offset_check_overflow::<A, _>(&dim, strides).unwrap();
                } else {
                    dimension::size_of_shape_checked(&dim).unwrap();
                }
            }
            let strides = shape.strides.strides_for_dim(&dim);
            RawArrayViewMut::new_(ptr, dim, strides)
        }
        #[doc = " Converts to a non-mutable `RawArrayView`."]
        #[inline]
        pub(crate) fn into_raw_view(self) -> RawArrayView<A, D> {
            unsafe { RawArrayView::new(self.ptr, self.dim, self.strides) }
        }
        #[doc = " Converts to a mutable view of the array."]
        #[doc = ""]
        #[doc = " # Safety"]
        #[doc = ""]
        #[doc = " From a safety standpoint, this is equivalent to dereferencing a raw"]
        #[doc = " pointer for every element in the array. You must ensure that all of the"]
        #[doc = " data is valid, ensure that the pointer is aligned, and choose the"]
        #[doc = " correct lifetime."]
        #[inline]
        pub unsafe fn deref_into_view_mut<'a>(self) -> ArrayViewMut<'a, A, D> {
            debug_assert!(
                is_aligned(self.ptr.as_ptr()),
                "The pointer must be aligned."
            );
            ArrayViewMut::new(self.ptr, self.dim, self.strides)
        }
        #[doc = " Cast the raw pointer of the raw array view to a different type"]
        #[doc = ""]
        #[doc = " **Panics** if element size is not compatible."]
        #[doc = ""]
        #[doc = " Lack of panic does not imply it is a valid cast. The cast works the same"]
        #[doc = " way as regular raw pointer casts."]
        #[doc = ""]
        #[doc = " While this method is safe, for the same reason as regular raw pointer"]
        #[doc = " casts are safe, access through the produced raw view is only possible"]
        #[doc = " in an unsafe block or function."]
        pub fn cast<B>(self) -> RawArrayViewMut<B, D> {
            assert_eq!(
                mem::size_of::<B>(),
                mem::size_of::<A>(),
                "size mismatch in raw view cast"
            );
            let ptr = self.ptr.cast::<B>();
            unsafe { RawArrayViewMut::new(ptr, self.dim, self.strides) }
        }
    }
    impl<T, D> RawArrayViewMut<Complex<T>, D> where D: Dimension {}
}
mod impl_cow {
    use crate::imp_prelude::*;
    impl<'a, A, D> From<ArrayView<'a, A, D>> for CowArray<'a, A, D>
    where
        D: Dimension,
    {
        fn from(view: ArrayView<'a, A, D>) -> CowArray<'a, A, D> {
            unsafe {
                ArrayBase::from_data_ptr(CowRepr::View(view.data), view.ptr)
                    .with_strides_dim(view.strides, view.dim)
            }
        }
    }
    impl<'a, A, D> From<Array<A, D>> for CowArray<'a, A, D>
    where
        D: Dimension,
    {
        fn from(array: Array<A, D>) -> CowArray<'a, A, D> {
            unsafe {
                ArrayBase::from_data_ptr(CowRepr::Owned(array.data), array.ptr)
                    .with_strides_dim(array.strides, array.dim)
            }
        }
    }
}
#[doc = " Returns `true` if the pointer is aligned."]
pub(crate) fn is_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize) % ::std::mem::align_of::<T>() == 0
}
