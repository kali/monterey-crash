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
        unimplemented!()
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn Ix1(i0: Ix) -> Ix1 {
        Dim::new([i0])
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn Ix2(i0: Ix, i1: Ix) -> Ix2 {
        unimplemented!()
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn Ix3(i0: Ix, i1: Ix, i2: Ix) -> Ix3 {
        unimplemented!()
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn IxDyn(ix: &[Ix]) -> IxDyn {
        unimplemented!()
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
    use alloc::vec::Vec;
    use std::mem;
    use std::ops::{Index, IndexMut};
    use std::{hash, mem::size_of};
    use std::{iter::FromIterator, slice};
    #[cold]
    #[inline(never)]
    pub(crate) fn array_out_of_bounds() -> ! {
        unimplemented!()
    }
    #[inline(always)]
    pub fn debug_bounds_check<S, D, I>(_a: &ArrayBase<S, D>, _index: &I)
    where
        D: Dimension,
        I: NdIndex<D>,
        S: Data,
    {
        unimplemented!()
    }
    impl<A, S> From<Vec<A>> for ArrayBase<S, Ix1>
    where
        S: DataOwned<Elem = A>,
    {
        fn from(v: Vec<A>) -> Self {
            unimplemented!()
        }
    }
    impl<'a, A, Slice: ?Sized> From<&'a Slice> for ArrayView<'a, A, Ix1>
    where
        Slice: AsRef<[A]>,
    {
        fn from(slice: &'a Slice) -> Self {
            unimplemented!()
        }
    }
    impl<'a, A, const N: usize> From<&'a [[A; N]]> for ArrayView<'a, A, Ix2> {
        fn from(xs: &'a [[A; N]]) -> Self {
            unimplemented!()
        }
    }
    impl<'a, A, Slice: ?Sized> From<&'a mut Slice> for ArrayViewMut<'a, A, Ix1>
    where
        Slice: AsMut<[A]>,
    {
        fn from(slice: &'a mut Slice) -> Self {
            unimplemented!()
        }
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
            unimplemented!()
        }
        pub(crate) fn as_ptr(&self) -> *const A {
            self.ptr.as_ptr()
        }
        pub(crate) fn as_nonnull_mut(&mut self) -> NonNull<A> {
            self.ptr
        }
        pub(crate) fn as_end_nonnull(&self) -> NonNull<A> {
            unimplemented!()
        }
        #[must_use = "must use new pointer to update existing pointers"]
        pub(crate) fn reserve(&mut self, additional: usize) -> NonNull<A> {
            unimplemented!()
        }
        pub(crate) unsafe fn set_len(&mut self, new_len: usize) {
            unimplemented!()
        }
        fn modify_as_vec(&mut self, f: impl FnOnce(Vec<A>) -> Vec<A>) {
            unimplemented!()
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
                    unimplemented!()
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
            unimplemented!()
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
            unimplemented!()
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
            unimplemented!()
        }
        #[inline]
        #[allow(clippy::wrong_self_convention)]
        fn is_unique(&mut self) -> bool {
            unimplemented!()
        }
    }
    unsafe impl<A> RawData for RawViewRepr<*const A> {
        type Elem = A;
        #[inline]
        fn _data_slice(&self) -> Option<&[A]> {
            unimplemented!()
        }
        #[inline(always)]
        fn _is_pointer_inbounds(&self, _ptr: *const Self::Elem) -> bool {
            unimplemented!()
        }
        private_impl! {}
    }
    unsafe impl<A> RawData for RawViewRepr<*mut A> {
        type Elem = A;
        #[inline]
        fn _data_slice(&self) -> Option<&[A]> {
            unimplemented!()
        }
        #[inline(always)]
        fn _is_pointer_inbounds(&self, _ptr: *const Self::Elem) -> bool {
            unimplemented!()
        }
        private_impl! {}
    }
    unsafe impl<A> RawData for OwnedArcRepr<A> {
        type Elem = A;
        fn _data_slice(&self) -> Option<&[A]> {
            unimplemented!()
        }
        fn _is_pointer_inbounds(&self, self_ptr: *const Self::Elem) -> bool {
            unimplemented!()
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
            unimplemented!()
        }
        fn try_is_unique(&mut self) -> Option<bool> {
            unimplemented!()
        }
    }
    unsafe impl<A> Data for OwnedArcRepr<A> {
        fn into_owned<D>(mut self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
        where
            A: Clone,
            D: Dimension,
        {
            unimplemented!()
        }
        fn try_into_owned_nocopy<D>(
            self_: ArrayBase<Self, D>,
        ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
        where
            D: Dimension,
        {
            unimplemented!()
        }
    }
    unsafe impl<A> DataMut for OwnedArcRepr<A> where A: Clone {}
    unsafe impl<A> RawData for OwnedRepr<A> {
        type Elem = A;
        fn _data_slice(&self) -> Option<&[A]> {
            unimplemented!()
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
            unimplemented!()
        }
        #[inline]
        fn try_is_unique(&mut self) -> Option<bool> {
            unimplemented!()
        }
    }
    unsafe impl<A> Data for OwnedRepr<A> {
        #[inline]
        fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
        where
            A: Clone,
            D: Dimension,
        {
            unimplemented!()
        }
        #[inline]
        fn try_into_owned_nocopy<D>(
            self_: ArrayBase<Self, D>,
        ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
        where
            D: Dimension,
        {
            unimplemented!()
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
            unimplemented!()
        }
        #[inline(always)]
        fn _is_pointer_inbounds(&self, _ptr: *const Self::Elem) -> bool {
            unimplemented!()
        }
        private_impl! {}
    }
    unsafe impl<'a, A> Data for ViewRepr<&'a A> {
        fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
        where
            Self::Elem: Clone,
            D: Dimension,
        {
            unimplemented!()
        }
        fn try_into_owned_nocopy<D>(
            self_: ArrayBase<Self, D>,
        ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
        where
            D: Dimension,
        {
            unimplemented!()
        }
    }
    unsafe impl<'a, A> RawDataClone for ViewRepr<&'a A> {
        unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
            unimplemented!()
        }
    }
    unsafe impl<'a, A> RawData for ViewRepr<&'a mut A> {
        type Elem = A;
        #[inline]
        fn _data_slice(&self) -> Option<&[A]> {
            unimplemented!()
        }
        #[inline(always)]
        fn _is_pointer_inbounds(&self, _ptr: *const Self::Elem) -> bool {
            unimplemented!()
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
            unimplemented!()
        }
        #[inline]
        fn try_is_unique(&mut self) -> Option<bool> {
            unimplemented!()
        }
    }
    unsafe impl<'a, A> Data for ViewRepr<&'a mut A> {
        fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
        where
            Self::Elem: Clone,
            D: Dimension,
        {
            unimplemented!()
        }
        fn try_into_owned_nocopy<D>(
            self_: ArrayBase<Self, D>,
        ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
        where
            D: Dimension,
        {
            unimplemented!()
        }
    }
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
            unimplemented!()
        }
    }
    unsafe impl<'a, A> RawData for CowRepr<'a, A> {
        type Elem = A;
        fn _data_slice(&self) -> Option<&[A]> {
            unimplemented!()
        }
        #[inline]
        fn _is_pointer_inbounds(&self, ptr: *const Self::Elem) -> bool {
            unimplemented!()
        }
        private_impl! {}
    }
    pub trait RawDataSubst<A>: RawData {
        type Output: RawData<Elem = A>;
        unsafe fn data_subst(self) -> Self::Output;
    }
    impl<A, B> RawDataSubst<B> for OwnedRepr<A> {
        type Output = OwnedRepr<B>;
        unsafe fn data_subst(self) -> Self::Output {
            unimplemented!()
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
    impl<A, const N: usize, const M: usize> From<Vec<[[A; M]; N]>> for Array3<A> {
        fn from(mut xs: Vec<[[A; M]; N]>) -> Self {
            unimplemented!()
        }
    }
}
pub use crate::iterators::iter;
mod error {
    use super::Dimension;
    use std::fmt;
    #[derive(Clone)]
    pub struct ShapeError {
        repr: ErrorKind,
    }
    impl ShapeError {
        #[inline]
        pub fn kind(&self) -> ErrorKind {
            unimplemented!()
        }
        pub fn from_kind(error: ErrorKind) -> Self {
            unimplemented!()
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
        unimplemented!()
    }
    impl PartialEq for ErrorKind {
        #[inline(always)]
        fn eq(&self, rhs: &Self) -> bool {
            unimplemented!()
        }
    }
    impl fmt::Display for ShapeError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            unimplemented!()
        }
    }
    impl fmt::Debug for ShapeError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            unimplemented!()
        }
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
            unimplemented!()
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
        unimplemented!()
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
            let index = if sz != 0 {
                Some(self.start)
            } else {
                unimplemented!()
            };
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
            unimplemented!()
        }
        private_impl! {}
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
        unimplemented!()
    }
    impl<D> Iterator for IndicesIterF<D>
    where
        D: Dimension,
    {
        type Item = D::Pattern;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            unimplemented!()
        }
    }
    impl<D> ExactSizeIterator for IndicesIterF<D> where D: Dimension {}
}
mod iterators {
    #[macro_use]
    mod macros {
        macro_rules ! impl_ndproducer { ([$ ($ typarm : tt) *] [Clone => $ ($ cloneparm : tt) *] $ typename : ident { $ base : ident , $ ($ fieldname : ident ,) * } $ fulltype : ty { $ (type $ atyn : ident = $ atyv : ty ;) * unsafe fn item (&$ self_ : ident , $ ptr : pat) { $ refexpr : expr } }) => { impl <$ ($ typarm) *> NdProducer for $ fulltype { $ (type $ atyn = $ atyv ;) * type Ptr = * mut A ; type Stride = isize ; fn raw_dim (& self) -> D { self .$ base . raw_dim () } fn layout (& self) -> Layout { self .$ base . layout () } fn as_ptr (& self) -> * mut A { self .$ base . as_ptr () as * mut _ } fn contiguous_stride (& self) -> isize { self .$ base . contiguous_stride () } unsafe fn as_ref (&$ self_ , $ ptr : * mut A) -> Self :: Item { $ refexpr } unsafe fn uget_ptr (& self , i : & Self :: Dim) -> * mut A { self .$ base . uget_ptr (i) } fn stride_of (& self , axis : Axis) -> isize { self .$ base . stride_of (axis) } fn split_at (self , axis : Axis , index : usize) -> (Self , Self) { let (a , b) = self .$ base . split_at (axis , index) ; ($ typename { $ base : a , $ ($ fieldname : self .$ fieldname . clone () ,) * } , $ typename { $ base : b , $ ($ fieldname : self .$ fieldname ,) * }) } private_impl ! { } } expand_if ! (@ nonempty [$ ($ cloneparm) *] impl <$ ($ cloneparm) *> Clone for $ fulltype { fn clone (& self) -> Self { $ typename { $ base : self . base . clone () , $ ($ fieldname : self .$ fieldname . clone () ,) * } } }) ; } }
    }
    mod chunks {
        use crate::imp_prelude::*;
        use crate::ElementsBase;
        use crate::ElementsBaseMut;
        type BaseProducerRef<'a, A, D> = ArrayView<'a, A, D>;
        type BaseProducerMut<'a, A, D> = ArrayViewMut<'a, A, D>;
        pub struct ExactChunks<'a, A, D> {
            base: BaseProducerRef<'a, A, D>,
            chunk: D,
            inner_strides: D,
        }
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
        pub struct ExactChunksIterMut<'a, A, D> {
            iter: ElementsBaseMut<'a, A, D>,
            chunk: D,
            inner_strides: D,
        }
    }
    mod into_iter {
        use super::Baseiter;
        use crate::imp_prelude::*;
        use crate::OwnedRepr;
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
        use crate::imp_prelude::*;
        use crate::{Layout, NdProducer};
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
                unimplemented!()
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
                unimplemented!()
            }
        }
    }
    mod windows {
        use super::ElementsBase;
        use crate::imp_prelude::*;
        pub struct Windows<'a, A, D> {
            base: ArrayView<'a, A, D>,
            window: D,
            strides: D,
        }
    }
    pub use self::chunks::{ExactChunks, ExactChunksIter, ExactChunksIterMut, ExactChunksMut};
    pub use self::lanes::{Lanes, LanesMut};
    pub use self::windows::Windows;
    use super::{ArrayBase, ArrayView, ArrayViewMut, Axis, Data, NdProducer, RemoveAxis};
    use super::{Dimension, Ix, Ixs};
    use crate::Ix1;
    use alloc::vec::Vec;
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
            unimplemented!()
        }
    }
    impl<A, D: Dimension> Iterator for Baseiter<A, D> {
        type Item = *mut A;
        #[inline]
        fn next(&mut self) -> Option<*mut A> {
            unimplemented!()
        }
    }
    impl<A> DoubleEndedIterator for Baseiter<A, Ix1> {
        #[inline]
        fn next_back(&mut self) -> Option<*mut A> {
            unimplemented!()
        }
    }
    clone_bounds ! ([A , D : Clone] Baseiter [A , D] { @ copy { ptr , } dim , strides , index , });
    clone_bounds ! (['a , A , D : Clone] ElementsBase ['a , A , D] { @ copy { life , } inner , });
    impl<'a, A, D: Dimension> ElementsBase<'a, A, D> {
        pub fn new(v: ArrayView<'a, A, D>) -> Self {
            unimplemented!()
        }
    }
    impl<'a, A, D: Dimension> Iterator for ElementsBase<'a, A, D> {
        type Item = &'a A;
        #[inline]
        fn next(&mut self) -> Option<&'a A> {
            unimplemented!()
        }
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
            unimplemented!()
        }
    }
    impl<'a, A, D> IterMut<'a, A, D>
    where
        D: Dimension,
    {
        pub(crate) fn new(self_: ArrayViewMut<'a, A, D>) -> Self {
            unimplemented!()
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
            unimplemented!()
        }
    }
    #[derive(Clone)]
    pub struct IndexedIter<'a, A, D>(ElementsBase<'a, A, D>);
    pub struct IndexedIterMut<'a, A, D>(ElementsBaseMut<'a, A, D>);
    impl<'a, A, D: Dimension> Iterator for Iter<'a, A, D> {
        type Item = &'a A;
        #[inline]
        fn next(&mut self) -> Option<&'a A> {
            unimplemented!()
        }
    }
    impl<'a, A, D> ExactSizeIterator for Iter<'a, A, D> where D: Dimension {}
    impl<'a, A, D: Dimension> Iterator for IterMut<'a, A, D> {
        type Item = &'a mut A;
        #[inline]
        fn next(&mut self) -> Option<&'a mut A> {
            unimplemented!()
        }
    }
    impl<'a, A, D> ExactSizeIterator for IterMut<'a, A, D> where D: Dimension {}
    impl<'a, A, D: Dimension> Iterator for ElementsBaseMut<'a, A, D> {
        type Item = &'a mut A;
        #[inline]
        fn next(&mut self) -> Option<&'a mut A> {
            unimplemented!()
        }
    }
    pub struct LanesIter<'a, A, D> {
        inner_len: Ix,
        inner_stride: Ixs,
        iter: Baseiter<A, D>,
        life: PhantomData<&'a A>,
    }
    pub struct LanesIterMut<'a, A, D> {
        inner_len: Ix,
        inner_stride: Ixs,
        iter: Baseiter<A, D>,
        life: PhantomData<&'a mut A>,
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
    impl<A, D: Dimension> AxisIterCore<A, D> {
        fn new<S, Di>(v: ArrayBase<S, Di>, axis: Axis) -> Self
        where
            Di: RemoveAxis<Smaller = D>,
            S: Data<Elem = A>,
        {
            unimplemented!()
        }
        #[inline]
        unsafe fn offset(&self, index: usize) -> *mut A {
            unimplemented!()
        }
        fn split_at(self, index: usize) -> (Self, Self) {
            unimplemented!()
        }
    }
    impl<A, D> Iterator for AxisIterCore<A, D>
    where
        D: Dimension,
    {
        type Item = *mut A;
        fn next(&mut self) -> Option<Self::Item> {
            unimplemented!()
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
            unimplemented!()
        }
        pub fn split_at(self, index: usize) -> (Self, Self) {
            unimplemented!()
        }
    }
    impl<'a, A, D> Iterator for AxisIter<'a, A, D>
    where
        D: Dimension,
    {
        type Item = ArrayView<'a, A, D>;
        fn next(&mut self) -> Option<Self::Item> {
            unimplemented!()
        }
    }
    impl<'a, A, D> ExactSizeIterator for AxisIter<'a, A, D> where D: Dimension {}
    pub struct AxisIterMut<'a, A, D> {
        iter: AxisIterCore<A, D>,
        life: PhantomData<&'a mut A>,
    }
    impl<'a, A, D: Dimension> AxisIterMut<'a, A, D> {
        pub fn split_at(self, index: usize) -> (Self, Self) {
            unimplemented!()
        }
    }
    impl<'a, A, D> Iterator for AxisIterMut<'a, A, D>
    where
        D: Dimension,
    {
        type Item = ArrayViewMut<'a, A, D>;
        fn next(&mut self) -> Option<Self::Item> {
            unimplemented!()
        }
    }
    impl<'a, A, D> ExactSizeIterator for AxisIterMut<'a, A, D> where D: Dimension {}
    impl<'a, A, D: Dimension> NdProducer for AxisIter<'a, A, D> {
        type Item = <Self as Iterator>::Item;
        type Dim = Ix1;
        type Ptr = *mut A;
        type Stride = isize;
        fn layout(&self) -> crate::Layout {
            unimplemented!()
        }
        fn raw_dim(&self) -> Self::Dim {
            unimplemented!()
        }
        fn as_ptr(&self) -> Self::Ptr {
            unimplemented!()
        }
        fn contiguous_stride(&self) -> isize {
            unimplemented!()
        }
        unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
            unimplemented!()
        }
        unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
            unimplemented!()
        }
        fn stride_of(&self, _axis: Axis) -> isize {
            unimplemented!()
        }
        fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
            unimplemented!()
        }
        private_impl! {}
    }
    impl<'a, A, D: Dimension> NdProducer for AxisIterMut<'a, A, D> {
        type Item = <Self as Iterator>::Item;
        type Dim = Ix1;
        type Ptr = *mut A;
        type Stride = isize;
        fn layout(&self) -> crate::Layout {
            unimplemented!()
        }
        fn raw_dim(&self) -> Self::Dim {
            unimplemented!()
        }
        fn as_ptr(&self) -> Self::Ptr {
            unimplemented!()
        }
        fn contiguous_stride(&self) -> isize {
            unimplemented!()
        }
        unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
            unimplemented!()
        }
        unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
            unimplemented!()
        }
        fn stride_of(&self, _axis: Axis) -> isize {
            unimplemented!()
        }
        fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
            unimplemented!()
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
        unimplemented!()
    }
    pub struct AxisChunksIterMut<'a, A, D> {
        iter: AxisIterCore<A, D>,
        partial_chunk_index: usize,
        partial_chunk_dim: D,
        life: PhantomData<&'a mut A>,
    }
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait TrustedIterator {}
    use crate::indexes::IndicesIterF;
    use crate::iter::IndicesIter;
    #[cfg(feature = "std")]
    use crate::{geomspace::Geomspace, linspace::Linspace, logspace::Logspace};
    unsafe impl<'a, A, D> TrustedIterator for Iter<'a, A, D> {}
    unsafe impl<'a, A, D> TrustedIterator for IterMut<'a, A, D> {}
    unsafe impl<'a, A> TrustedIterator for slice::Iter<'a, A> {}
    unsafe impl<'a, A> TrustedIterator for slice::IterMut<'a, A> {}
    unsafe impl TrustedIterator for ::std::ops::Range<usize> {}
    unsafe impl<D> TrustedIterator for IndicesIter<D> where D: Dimension {}
    unsafe impl<D> TrustedIterator for IndicesIterF<D> where D: Dimension {}
    pub fn to_vec<I>(iter: I) -> Vec<I::Item>
    where
        I: TrustedIterator + ExactSizeIterator,
    {
        unimplemented!()
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
                unimplemented!()
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
            unimplemented!()
        }
        #[inline(always)]
        pub(crate) fn intersect(self, other: Layout) -> Layout {
            unimplemented!()
        }
        #[inline(always)]
        pub(crate) fn also(self, other: Layout) -> Layout {
            unimplemented!()
        }
        #[inline(always)]
        pub(crate) fn one_dimensional() -> Layout {
            unimplemented!()
        }
        #[inline(always)]
        pub(crate) fn c() -> Layout {
            unimplemented!()
        }
        #[inline(always)]
        pub(crate) fn f() -> Layout {
            unimplemented!()
        }
        #[inline(always)]
        pub(crate) fn cpref() -> Layout {
            unimplemented!()
        }
        #[inline(always)]
        pub(crate) fn fpref() -> Layout {
            unimplemented!()
        }
        #[inline(always)]
        pub(crate) fn none() -> Layout {
            unimplemented!()
        }
        #[inline]
        pub(crate) fn tendency(self) -> i32 {
            unimplemented!()
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
}
mod math_cell {
    use std::cell::Cell;
    use std::cmp::Ordering;
    use std::ops::{Deref, DerefMut};
    #[repr(transparent)]
    #[derive(Default)]
    pub struct MathCell<T>(Cell<T>);
    impl<T> Deref for MathCell<T> {
        type Target = Cell<T>;
        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            unimplemented!()
        }
    }
    impl<T> PartialEq for MathCell<T>
    where
        T: Copy + PartialEq,
    {
        fn eq(&self, rhs: &Self) -> bool {
            unimplemented!()
        }
    }
}
mod numeric_util {
    use std::cmp;
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
    #[must_use]
    pub(crate) struct Partial<T> {
        ptr: *mut T,
        pub(crate) len: usize,
    }
    impl<T> Partial<T> {
        pub(crate) unsafe fn new(ptr: *mut T) -> Self {
            unimplemented!()
        }
        pub(crate) fn release_ownership(mut self) -> usize {
            unimplemented!()
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
                    unimplemented!()
                }
            }
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
            let st = if shape.is_c() {
                Strides::C
            } else {
                unimplemented!()
            };
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
            unimplemented!()
        }
        fn set_f(self, is_f: bool) -> Shape<Self::Dim> {
            unimplemented!()
        }
        fn strides(self, st: T) -> StrideShape<Self::Dim> {
            unimplemented!()
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
            unimplemented!()
        }
        fn set_f(mut self, is_f: bool) -> Self {
            unimplemented!()
        }
        fn strides(self, st: D) -> StrideShape<D> {
            unimplemented!()
        }
    }
    pub trait ShapeArg {
        type Dim: Dimension;
        fn into_shape_and_order(self) -> (Self::Dim, Option<Order>);
    }
}
#[macro_use]
mod slice {
    use crate::error::{ErrorKind, ShapeError};
    use crate::{ArrayViewMut, DimAdd, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
    use std::marker::PhantomData;
    use std::ops::{Deref, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Slice {
        pub start: isize,
        pub end: Option<isize>,
        pub step: isize,
    }
    impl Slice {}
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
            unimplemented!()
        }
        pub fn is_new_axis(&self) -> bool {
            unimplemented!()
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
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait SliceArg<D: Dimension>: AsRef<[SliceInfoElem]> {
        type OutDim: Dimension;
        fn in_ndim(&self) -> usize;
        fn out_ndim(&self) -> usize;
        private_decl! {}
    }
    unsafe impl SliceArg<IxDyn> for [SliceInfoElem] {
        type OutDim = IxDyn;
        fn in_ndim(&self) -> usize {
            unimplemented!()
        }
        fn out_ndim(&self) -> usize {
            unimplemented!()
        }
        private_impl! {}
    }
    #[derive(Debug)]
    pub struct SliceInfo<T, Din: Dimension, Dout: Dimension> {
        in_dim: PhantomData<Din>,
        out_dim: PhantomData<Dout>,
        indices: T,
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
            unimplemented!()
        }
        fn next_out_dim<D>(
            &self,
            _: PhantomData<D>,
        ) -> PhantomData<<D as DimAdd<Self::OutDim>>::Output>
        where
            D: Dimension + DimAdd<Self::OutDim>,
        {
            unimplemented!()
        }
    }
    pub trait MultiSliceArg<'a, A, D>
    where
        A: 'a,
        D: Dimension,
    {
        type Output;
        fn multi_slice_move(&self, view: ArrayViewMut<'a, A, D>) -> Self::Output;
        private_decl! {}
    }
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
            unimplemented!()
        }
    }
    impl<D> SplitAt for D
    where
        D: Dimension,
    {
        fn split_at(self, axis: Axis, index: Ix) -> (Self, Self) {
            unimplemented!()
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
        unimplemented!()
    }
}
mod low_level_util {
    #[must_use]
    pub(crate) struct AbortIfPanic(pub(crate) &'static &'static str);
    impl AbortIfPanic {
        #[inline]
        pub(crate) fn defuse(self) {
            unimplemented!()
        }
    }
}
#[macro_use]
mod zip {
    mod ndproducer {
        use crate::imp_prelude::*;
        use crate::Layout;
        use crate::NdIndex;
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
                unimplemented!()
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
                unimplemented!()
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
                unimplemented!()
            }
            private_impl! {}
        }
        impl<T> Offset for *mut T {
            type Stride = isize;
            unsafe fn stride_offset(self, s: Self::Stride, index: usize) -> Self {
                unimplemented!()
            }
            private_impl! {}
        }
        impl<'a, A, D: Dimension> NdProducer for ArrayView<'a, A, D> {
            type Item = &'a A;
            type Dim = D;
            type Ptr = *mut A;
            type Stride = isize;
            private_impl! {}
            fn raw_dim(&self) -> Self::Dim {
                unimplemented!()
            }
            fn as_ptr(&self) -> *mut A {
                unimplemented!()
            }
            fn layout(&self) -> Layout {
                unimplemented!()
            }
            unsafe fn as_ref(&self, ptr: *mut A) -> Self::Item {
                unimplemented!()
            }
            unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
                unimplemented!()
            }
            fn stride_of(&self, axis: Axis) -> isize {
                unimplemented!()
            }
            #[inline(always)]
            fn contiguous_stride(&self) -> Self::Stride {
                unimplemented!()
            }
            fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
                unimplemented!()
            }
        }
        impl<'a, A, D: Dimension> NdProducer for ArrayViewMut<'a, A, D> {
            type Item = &'a mut A;
            type Dim = D;
            type Ptr = *mut A;
            type Stride = isize;
            private_impl! {}
            fn raw_dim(&self) -> Self::Dim {
                unimplemented!()
            }
            fn as_ptr(&self) -> *mut A {
                unimplemented!()
            }
            fn layout(&self) -> Layout {
                unimplemented!()
            }
            unsafe fn as_ref(&self, ptr: *mut A) -> Self::Item {
                unimplemented!()
            }
            unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
                unimplemented!()
            }
            fn stride_of(&self, axis: Axis) -> isize {
                unimplemented!()
            }
            #[inline(always)]
            fn contiguous_stride(&self) -> Self::Stride {
                unimplemented!()
            }
            fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
                unimplemented!()
            }
        }
        impl<A, D: Dimension> NdProducer for RawArrayViewMut<A, D> {
            type Item = *mut A;
            type Dim = D;
            type Ptr = *mut A;
            type Stride = isize;
            private_impl! {}
            fn raw_dim(&self) -> Self::Dim {
                unimplemented!()
            }
            fn as_ptr(&self) -> *mut A {
                unimplemented!()
            }
            fn layout(&self) -> Layout {
                unimplemented!()
            }
            unsafe fn as_ref(&self, ptr: *mut A) -> *mut A {
                unimplemented!()
            }
            unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
                unimplemented!()
            }
            fn stride_of(&self, axis: Axis) -> isize {
                unimplemented!()
            }
            #[inline(always)]
            fn contiguous_stride(&self) -> Self::Stride {
                unimplemented!()
            }
            fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
                unimplemented!()
            }
        }
    }
    pub use self::ndproducer::{IntoNdProducer, NdProducer, Offset};
    use crate::dimension;
    use crate::imp_prelude::*;
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
        unimplemented!()
    }
    impl<S, D> ArrayBase<S, D>
    where
        S: RawData,
        D: Dimension,
    {
        pub(crate) fn layout_impl(&self) -> Layout {
            unimplemented!()
        }
    }
    impl<'a, A, D, E> Broadcast<E> for ArrayView<'a, A, D>
    where
        E: IntoDimension,
        D: Dimension,
    {
        type Output = ArrayView<'a, A, E::Dim>;
        fn broadcast_unwrap(self, shape: E) -> Self::Output {
            unimplemented!()
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
            unimplemented!()
        }
    }
    #[inline]
    fn zip_dimension_check<D, P>(dimension: &D, part: &P)
    where
        D: Dimension,
        P: NdProducer<Dim = D>,
    {
        unimplemented!()
    }
    impl<Parts, D> Zip<Parts, D>
    where
        D: Dimension,
    {
        pub fn size(&self) -> usize {
            unimplemented!()
        }
        fn len_of(&self, axis: Axis) -> usize {
            unimplemented!()
        }
        fn prefer_f(&self) -> bool {
            unimplemented!()
        }
        fn max_stride_axis(&self) -> Axis {
            unimplemented!()
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
            unimplemented!()
        }
        fn for_each_core_contiguous<F, Acc>(&mut self, acc: Acc, mut function: F) -> FoldWhile<Acc>
        where
            F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
            P: ZippableTuple<Dim = D>,
        {
            unimplemented!()
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
            unimplemented!()
        }
        fn for_each_core_strided<F, Acc>(&mut self, acc: Acc, function: F) -> FoldWhile<Acc>
        where
            F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
            P: ZippableTuple<Dim = D>,
        {
            unimplemented!()
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
            unimplemented!()
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
            unimplemented!()
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
            unimplemented!()
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
            unimplemented!()
        }
        pub fn is_done(&self) -> bool {
            unimplemented!()
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
    }
    mod axes {
        use crate::{Axis, Dimension, Ix, Ixs};
        pub(crate) fn axes_of<'a, D>(d: &'a D, strides: &'a D) -> Axes<'a, D>
        where
            D: Dimension,
        {
            unimplemented!()
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
                unimplemented!()
            }
        }
        impl<'a, D> DoubleEndedIterator for Axes<'a, D>
        where
            D: Dimension,
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                unimplemented!()
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
                unimplemented!()
            }
            #[inline(always)]
            fn post_dec(&mut self) -> Self {
                unimplemented!()
            }
            #[inline(always)]
            fn pre_dec(&mut self) -> Self {
                unimplemented!()
            }
        }
    }
    mod axis {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct Axis(pub usize);
        impl Axis {
            #[inline(always)]
            pub fn index(self) -> usize {
                unimplemented!()
            }
        }
    }
    pub(crate) mod broadcast {
        use crate::error::*;
        use crate::{Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
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
                unimplemented!()
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
                unimplemented!()
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
                unimplemented!()
            }
            fn slice(&self) -> &[Ix];
            fn slice_mut(&mut self) -> &mut [Ix];
            fn as_array_view(&self) -> ArrayView1<'_, Ix> {
                unimplemented!()
            }
            fn as_array_view_mut(&mut self) -> ArrayViewMut1<'_, Ix> {
                unimplemented!()
            }
            fn equal(&self, rhs: &Self) -> bool {
                unimplemented!()
            }
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
            fn fortran_strides(&self) -> Self {
                unimplemented!()
            }
            fn zeros(ndim: usize) -> Self;
            #[inline]
            fn first_index(&self) -> Option<Self> {
                unimplemented!()
            }
            #[inline]
            fn next_for(&self, index: Self) -> Option<Self> {
                let mut index = index;
                let mut done = false;
                for (&dim, ix) in zip(self.slice(), index.slice_mut()).rev() {
                    *ix += 1;
                    if *ix == dim {
                        *ix = 0;
                    } else {
                        unimplemented!()
                    }
                }
                if done {
                    unimplemented!()
                } else {
                    None
                }
            }
            #[inline]
            fn next_for_f(&self, index: &mut Self) -> bool {
                unimplemented!()
            }
            fn strides_equivalent<D>(&self, strides1: &Self, strides2: &D) -> bool
            where
                D: Dimension,
            {
                unimplemented!()
            }
            fn stride_offset(index: &Self, strides: &Self) -> isize {
                unimplemented!()
            }
            fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize> {
                unimplemented!()
            }
            fn last_elem(&self) -> usize {
                unimplemented!()
            }
            fn set_last_elem(&mut self, i: usize) {
                unimplemented!()
            }
            fn is_contiguous(dim: &Self, strides: &Self) -> bool {
                unimplemented!()
            }
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
            fn min_stride_axis(&self, strides: &Self) -> Axis {
                unimplemented!()
            }
            fn max_stride_axis(&self, strides: &Self) -> Axis {
                unimplemented!()
            }
            fn into_dyn(self) -> IxDyn {
                unimplemented!()
            }
            fn from_dimension<D2: Dimension>(d: &D2) -> Option<Self> {
                unimplemented!()
            }
            fn insert_axis(&self, axis: Axis) -> Self::Larger;
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
                unimplemented!()
            }
            #[inline]
            fn slice(&self) -> &[Ix] {
                unimplemented!()
            }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] {
                unimplemented!()
            }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {
                unimplemented!()
            }
            #[inline]
            fn zeros(ndim: usize) -> Self {
                unimplemented!()
            }
            impl_insert_axis_array!(0);
            #[inline]
            fn try_remove_axis(&self, _ignore: Axis) -> Self::Smaller {
                unimplemented!()
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
                unimplemented!()
            }
            #[inline]
            fn slice(&self) -> &[Ix] {
                unimplemented!()
            }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] {
                unimplemented!()
            }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {
                unimplemented!()
            }
            #[inline]
            fn zeros(ndim: usize) -> Self {
                unimplemented!()
            }
            impl_insert_axis_array!(1);
            #[inline]
            fn try_remove_axis(&self, axis: Axis) -> Self::Smaller {
                unimplemented!()
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
                unimplemented!()
            }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {
                unimplemented!()
            }
            #[inline]
            fn slice(&self) -> &[Ix] {
                unimplemented!()
            }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] {
                unimplemented!()
            }
            #[inline]
            fn zeros(ndim: usize) -> Self {
                unimplemented!()
            }
            impl_insert_axis_array!(2);
            #[inline]
            fn try_remove_axis(&self, axis: Axis) -> Self::Smaller {
                unimplemented!()
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
                unimplemented!()
            }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {
                unimplemented!()
            }
            #[inline]
            fn slice(&self) -> &[Ix] {
                unimplemented!()
            }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] {
                unimplemented!()
            }
            #[inline]
            fn zeros(ndim: usize) -> Self {
                unimplemented!()
            }
            impl_insert_axis_array!(3);
            #[inline]
            fn try_remove_axis(&self, axis: Axis) -> Self::Smaller {
                unimplemented!()
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
                unimplemented!()
            }
            #[inline]
            fn try_remove_axis(&self, axis: Axis) -> Self::Smaller {
                unimplemented!()
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
                unimplemented!()
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
        impl Default for IxDynRepr<Ix> {
            fn default() -> Self {
                unimplemented!()
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
                    unimplemented!()
                }
            }
        }
        impl<T: Copy + Zero> IxDynRepr<T> {
            fn from_vec_auto(v: Vec<T>) -> Self {
                if v.len() <= CAP {
                    Self::copy_from(&v)
                } else {
                    unimplemented!()
                }
            }
        }
        impl<T: Copy> IxDynRepr<T> {
            fn from_vec(v: Vec<T>) -> Self {
                unimplemented!()
            }
            fn from(x: &[T]) -> Self {
                unimplemented!()
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
                unimplemented!()
            }
        }
        impl<T: Hash> Hash for IxDynRepr<T> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                unimplemented!()
            }
        }
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
        pub struct IxDynImpl(IxDynRepr<Ix>);
        impl IxDynImpl {
            pub(crate) fn insert(&self, i: usize) -> Self {
                unimplemented!()
            }
            fn remove(&self, i: usize) -> Self {
                unimplemented!()
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
                unimplemented!()
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
        impl RemoveAxis for Dim<IxDynImpl> {
            fn remove_axis(&self, axis: Axis) -> Self {
                unimplemented!()
            }
        }
        impl IxDyn {
            #[inline]
            pub fn zeros(n: usize) -> IxDyn {
                const ZEROS: &[usize] = &[0; 4];
                if n <= ZEROS.len() {
                    Dim(&ZEROS[..n])
                } else {
                    unimplemented!()
                }
            }
        }
    }
    mod ndindex {
        use super::{stride_offset, stride_offset_checked};
        use crate::{
            Dim, Dimension, IntoDimension, Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, IxDynImpl,
        };
        use std::fmt::Debug;
        #[allow(clippy::missing_safety_doc)]
        pub unsafe trait NdIndex<E>: Debug {
            fn index_checked(&self, dim: &E, strides: &E) -> Option<isize>;
            fn index_unchecked(&self, strides: &E) -> isize;
        }
        unsafe impl<D> NdIndex<D> for D
        where
            D: Dimension,
        {
            fn index_checked(&self, dim: &D, strides: &D) -> Option<isize> {
                unimplemented!()
            }
            fn index_unchecked(&self, strides: &D) -> isize {
                unimplemented!()
            }
        }
        impl<'a> IntoDimension for &'a [Ix] {
            type Dim = IxDyn;
            fn into_dimension(self) -> Self::Dim {
                Dim(IxDynImpl::from(self))
            }
        }
    }
    mod ops {
        use crate::imp_prelude::*;
        pub trait DimAdd<D: Dimension> {
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
        pub trait RemoveAxis: Dimension {
            fn remove_axis(&self, axis: Axis) -> Self::Smaller;
        }
        impl RemoveAxis for Dim<[Ix; 1]> {
            #[inline]
            fn remove_axis(&self, axis: Axis) -> Ix0 {
                unimplemented!()
            }
        }
        impl RemoveAxis for Dim<[Ix; 2]> {
            #[inline]
            fn remove_axis(&self, axis: Axis) -> Ix1 {
                unimplemented!()
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
            unimplemented!()
        }
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
            unimplemented!()
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
                unimplemented!()
            }
        }
        impl<D> Index<usize> for Forward<&mut D>
        where
            D: Dimension,
        {
            type Output = usize;
            #[inline]
            fn index(&self, index: usize) -> &usize {
                unimplemented!()
            }
        }
        impl<D> IndexMut<usize> for Forward<&mut D>
        where
            D: Dimension,
        {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut usize {
                unimplemented!()
            }
        }
        impl<D> Index<usize> for Reverse<&D>
        where
            D: Dimension,
        {
            type Output = usize;
            #[inline]
            fn index(&self, index: usize) -> &usize {
                unimplemented!()
            }
        }
        impl<D> Index<usize> for Reverse<&mut D>
        where
            D: Dimension,
        {
            type Output = usize;
            #[inline]
            fn index(&self, index: usize) -> &usize {
                unimplemented!()
            }
        }
        impl<D> IndexMut<usize> for Reverse<&mut D>
        where
            D: Dimension,
        {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut usize {
                unimplemented!()
            }
        }
        pub(in crate::dimension) trait Sequence: Index<usize> {
            fn len(&self) -> usize;
        }
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
                unimplemented!()
            }
        }
        impl<D> Sequence for Forward<&mut D>
        where
            D: Dimension,
        {
            #[inline]
            fn len(&self) -> usize {
                unimplemented!()
            }
        }
        impl<D> SequenceMut for Forward<&mut D> where D: Dimension {}
        impl<D> Sequence for Reverse<&D>
        where
            D: Dimension,
        {
            #[inline]
            fn len(&self) -> usize {
                unimplemented!()
            }
        }
        impl<D> Sequence for Reverse<&mut D>
        where
            D: Dimension,
        {
            #[inline]
            fn len(&self) -> usize {
                unimplemented!()
            }
        }
        impl<D> SequenceMut for Reverse<&mut D> where D: Dimension {}
    }
    #[inline(always)]
    pub fn stride_offset(n: Ix, stride: Ix) -> isize {
        unimplemented!()
    }
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
                    unimplemented!()
                }
            }
        }
        false
    }
    pub fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ShapeError> {
        let size_nonzero = dim
            .slice()
            .iter()
            .filter(|&&d| d != 0)
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
        if size_nonzero > ::std::isize::MAX as usize {
            unimplemented!()
        } else {
            Ok(dim.size())
        }
    }
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
            unimplemented!()
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
            unimplemented!()
        }
        let max_offset_bytes = max_offset
            .checked_mul(elem_size)
            .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
        if max_offset_bytes > isize::MAX as usize {
            unimplemented!()
        }
        Ok(max_offset)
    }
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
            unimplemented!()
        }
        if !is_empty && max_offset >= data_len {
            unimplemented!()
        }
        if !is_empty && dim_stride_overlap(dim, strides) {
            unimplemented!()
        }
        Ok(())
    }
    #[inline]
    pub fn stride_offset_checked(dim: &[Ix], strides: &[Ix], index: &[Ix]) -> Option<isize> {
        unimplemented!()
    }
    pub fn strides_non_negative<D>(strides: &D) -> Result<(), ShapeError>
    where
        D: Dimension,
    {
        unimplemented!()
    }
    pub trait DimensionExt {
        fn axis(&self, axis: Axis) -> Ix;
        fn set_axis(&mut self, axis: Axis, value: Ix);
    }
    impl<D> DimensionExt for D
    where
        D: Dimension,
    {
        #[inline]
        fn axis(&self, axis: Axis) -> Ix {
            unimplemented!()
        }
        #[inline]
        fn set_axis(&mut self, axis: Axis, value: Ix) {
            unimplemented!()
        }
    }
    pub fn do_collapse_axis<D: Dimension>(
        dims: &mut D,
        strides: &D,
        axis: usize,
        index: usize,
    ) -> isize {
        unimplemented!()
    }
    #[inline]
    pub fn abs_index(len: Ix, index: Ixs) -> Ix {
        unimplemented!()
    }
    fn to_abs_slice(axis_len: usize, slice: Slice) -> (usize, usize, isize) {
        unimplemented!()
    }
    pub fn offset_from_low_addr_ptr_to_logical_ptr<D: Dimension>(dim: &D, strides: &D) -> usize {
        let offset = izip!(dim.slice(), strides.slice()).fold(0, |_offset, (&d, &s)| {
            let s = s as isize;
            if s < 0 && d > 1 {
                unimplemented!()
            } else {
                _offset
            }
        });
        debug_assert!(offset >= 0);
        offset as usize
    }
    pub fn do_slice(dim: &mut usize, stride: &mut usize, slice: Slice) -> isize {
        unimplemented!()
    }
    fn extended_gcd(a: isize, b: isize) -> (isize, (isize, isize)) {
        unimplemented!()
    }
    fn solve_linear_diophantine_eq(a: isize, b: isize, c: isize) -> Option<(isize, isize)> {
        unimplemented!()
    }
    fn arith_seq_intersect(
        (min1, max1, step1): (isize, isize, isize),
        (min2, max2, step2): (isize, isize, isize),
    ) -> bool {
        unimplemented!()
    }
    fn slice_min_max(axis_len: usize, slice: Slice) -> Option<(usize, usize)> {
        unimplemented!()
    }
    pub(crate) fn is_layout_c<D: Dimension>(dim: &D, strides: &D) -> bool {
        unimplemented!()
    }
    pub(crate) fn is_layout_f<D: Dimension>(dim: &D, strides: &D) -> bool {
        unimplemented!()
    }
    pub fn merge_axes<D>(dim: &mut D, strides: &mut D, take: Axis, into: Axis) -> bool
    where
        D: Dimension,
    {
        unimplemented!()
    }
    pub fn move_min_stride_axis_to_last<D>(dim: &mut D, strides: &mut D)
    where
        D: Dimension,
    {
        unimplemented!()
    }
}
pub use crate::layout::Layout;
pub use crate::zip::{FoldWhile, IntoNdProducer, NdProducer, Zip};
mod imp_prelude {
    pub use crate::dimension::DimensionExt;
    pub use crate::prelude::*;
    pub use crate::{
        CowRepr, Data, DataMut, DataOwned, DataShared, Ix, Ixs, RawData, RawDataMut, RawViewRepr,
        RemoveAxis, ViewRepr,
    };
}
pub mod prelude {
    pub use crate::ShapeBuilder;
    pub use crate::{
        ArcArray, Array, ArrayBase, ArrayView, ArrayViewMut, CowArray, RawArrayView,
        RawArrayViewMut,
    };
    pub use crate::{Array0, Array1, Array2, Array3, Array4, Array5, Array6, ArrayD};
    pub use crate::{Axis, Dim, Dimension};
    pub use crate::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
}
pub type Ix = usize;
pub type Ixs = isize;
pub struct ArrayBase<S, D>
where
    S: RawData,
{
    data: S,
    ptr: std::ptr::NonNull<S::Elem>,
    dim: D,
    strides: D,
}
pub type ArcArray<A, D> = ArrayBase<OwnedArcRepr<A>, D>;
pub type Array<A, D> = ArrayBase<OwnedRepr<A>, D>;
pub type CowArray<'a, A, D> = ArrayBase<CowRepr<'a, A>, D>;
pub type ArrayView<'a, A, D> = ArrayBase<ViewRepr<&'a A>, D>;
pub type ArrayViewMut<'a, A, D> = ArrayBase<ViewRepr<&'a mut A>, D>;
pub type RawArrayView<A, D> = ArrayBase<RawViewRepr<*const A>, D>;
pub type RawArrayViewMut<A, D> = ArrayBase<RawViewRepr<*mut A>, D>;
pub use data_repr::OwnedRepr;
#[derive(Debug)]
pub struct OwnedArcRepr<A>(Arc<OwnedRepr<A>>);
impl<A> Clone for OwnedArcRepr<A> {
    fn clone(&self) -> Self {
        unimplemented!()
    }
}
#[derive(Copy, Clone)]
pub struct RawViewRepr<A> {
    ptr: PhantomData<A>,
}
impl<A> RawViewRepr<A> {
    #[inline(always)]
    fn new() -> Self {
        unimplemented!()
    }
}
#[derive(Copy, Clone)]
pub struct ViewRepr<A> {
    life: PhantomData<A>,
}
impl<A> ViewRepr<A> {
    #[inline(always)]
    fn new() -> Self {
        unimplemented!()
    }
}
pub enum CowRepr<'a, A> {
    View(ViewRepr<&'a A>),
    Owned(OwnedRepr<A>),
}
impl<'a, A> CowRepr<'a, A> {}
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
    #![allow(clippy::match_wild_err_arm)]
    use crate::dimension;
    use crate::dimension::offset_from_low_addr_ptr_to_logical_ptr;
    use crate::extension::nonnull::nonnull_from_vec_data;
    use crate::imp_prelude::*;
    use crate::indexes;
    use crate::indices;
    use crate::iterators::to_vec_mapped;
    use crate::iterators::TrustedIterator;
    use crate::StrideShape;
    use alloc::vec;
    use alloc::vec::Vec;
    use rawpointer::PointerExt;
    use std::mem;
    use std::mem::MaybeUninit;
    impl<S, A> ArrayBase<S, Ix1>
    where
        S: DataOwned<Elem = A>,
    {
        pub fn from_vec(v: Vec<A>) -> Self {
            unimplemented!()
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
    impl<S, A, D> ArrayBase<S, D>
    where
        S: DataOwned<Elem = A>,
        D: Dimension,
    {
        pub fn from_shape_simple_fn<Sh, F>(shape: Sh, mut f: F) -> Self
        where
            Sh: ShapeBuilder<Dim = D>,
            F: FnMut() -> A,
        {
            unimplemented!()
        }
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
                unimplemented!()
            }
        }
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
            unimplemented!()
        }
        pub fn uninit<Sh>(shape: Sh) -> ArrayBase<S::MaybeUninit, D>
        where
            Sh: ShapeBuilder<Dim = D>,
        {
            unimplemented!()
        }
        pub fn build_uninit<Sh, F>(shape: Sh, builder: F) -> ArrayBase<S::MaybeUninit, D>
        where
            Sh: ShapeBuilder<Dim = D>,
            F: FnOnce(ArrayViewMut<MaybeUninit<A>, D>),
        {
            unimplemented!()
        }
    }
}
mod impl_methods {
    use crate::dimension;
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
    use crate::zip::{IntoNdProducer, Zip};
    use crate::{arraytraits, DimMax};
    use crate::{NdIndex, Slice, SliceInfoElem};
    use alloc::slice;
    use rawpointer::PointerExt;
    use std::mem::{size_of, ManuallyDrop};
    impl<A, S, D> ArrayBase<S, D>
    where
        S: RawData<Elem = A>,
        D: Dimension,
    {
        pub fn len(&self) -> usize {
            unimplemented!()
        }
        pub fn len_of(&self, axis: Axis) -> usize {
            unimplemented!()
        }
        pub fn is_empty(&self) -> bool {
            unimplemented!()
        }
        pub fn ndim(&self) -> usize {
            unimplemented!()
        }
        pub fn dim(&self) -> D::Pattern {
            unimplemented!()
        }
        pub fn raw_dim(&self) -> D {
            unimplemented!()
        }
        pub fn shape(&self) -> &[usize] {
            unimplemented!()
        }
        pub fn strides(&self) -> &[isize] {
            unimplemented!()
        }
        pub fn stride_of(&self, axis: Axis) -> isize {
            unimplemented!()
        }
        pub fn view(&self) -> ArrayView<'_, A, D>
        where
            S: Data,
        {
            unimplemented!()
        }
        pub fn view_mut(&mut self) -> ArrayViewMut<'_, A, D>
        where
            S: DataMut,
        {
            unimplemented!()
        }
        pub fn to_owned(&self) -> Array<A, D>
        where
            A: Clone,
            S: Data,
        {
            unimplemented!()
        }
        pub fn into_shared(self) -> ArcArray<A, D>
        where
            S: DataOwned,
        {
            unimplemented!()
        }
        pub fn iter(&self) -> Iter<'_, A, D>
        where
            S: Data,
        {
            unimplemented!()
        }
        pub fn iter_mut(&mut self) -> IterMut<'_, A, D>
        where
            S: DataMut,
        {
            unimplemented!()
        }
        fn get_0d(&self) -> &A
        where
            S: Data,
        {
            unimplemented!()
        }
        fn try_ensure_unique(&mut self)
        where
            S: RawDataMut,
        {
            unimplemented!()
        }
        fn ensure_unique(&mut self)
        where
            S: DataMut,
        {
            unimplemented!()
        }
        pub fn is_standard_layout(&self) -> bool {
            unimplemented!()
        }
        pub(crate) fn is_contiguous(&self) -> bool {
            unimplemented!()
        }
        #[inline(always)]
        pub fn as_ptr(&self) -> *const A {
            self.ptr.as_ptr() as *const A
        }
        #[inline]
        pub fn raw_view_mut(&mut self) -> RawArrayViewMut<A, D>
        where
            S: RawDataMut,
        {
            unimplemented!()
        }
        #[inline]
        pub(crate) unsafe fn raw_view_mut_unchecked(&mut self) -> RawArrayViewMut<A, D>
        where
            S: DataOwned,
        {
            unimplemented!()
        }
        pub fn as_slice_memory_order(&self) -> Option<&[A]>
        where
            S: Data,
        {
            unimplemented!()
        }
        pub fn as_slice_memory_order_mut(&mut self) -> Option<&mut [A]>
        where
            S: DataMut,
        {
            unimplemented!()
        }
        pub(crate) fn try_as_slice_memory_order_mut(&mut self) -> Result<&mut [A], &mut Self>
        where
            S: DataMut,
        {
            unimplemented!()
        }
        pub fn broadcast<E>(&self, dim: E) -> Option<ArrayView<'_, A, E::Dim>>
        where
            E: IntoDimension,
            S: Data,
        {
            unimplemented!()
        }
        pub fn axes(&self) -> Axes<'_, D> {
            unimplemented!()
        }
        pub fn invert_axis(&mut self, axis: Axis) {
            unimplemented!()
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
            unimplemented!()
        }
        #[inline(always)]
        fn zip_mut_with_by_rows<B, S2, E, F>(&mut self, rhs: &ArrayBase<S2, E>, mut f: F)
        where
            S: DataMut,
            S2: Data<Elem = B>,
            E: Dimension,
            F: FnMut(&mut A, &B),
        {
            unimplemented!()
        }
        fn zip_mut_with_elem<B, F>(&mut self, rhs_elem: &B, mut f: F)
        where
            S: DataMut,
            F: FnMut(&mut A, &B),
        {
            unimplemented!()
        }
        pub fn fold<'a, F, B>(&'a self, init: B, f: F) -> B
        where
            F: FnMut(B, &'a A) -> B,
            A: 'a,
            S: Data,
        {
            unimplemented!()
        }
        pub fn map<'a, B, F>(&'a self, f: F) -> Array<B, D>
        where
            F: FnMut(&'a A) -> B,
            A: 'a,
            S: Data,
        {
            unimplemented!()
        }
        pub fn map_inplace<'a, F>(&'a mut self, f: F)
        where
            S: DataMut,
            A: 'a,
            F: FnMut(&'a mut A),
        {
            unimplemented!()
        }
    }
    #[inline]
    unsafe fn unlimited_transmute<A, B>(data: A) -> B {
        unimplemented!()
    }
}
mod impl_owned_array {
    use crate::dimension;
    use crate::error::{ErrorKind, ShapeError};
    use crate::imp_prelude::*;
    use crate::iterators::Baseiter;
    use crate::low_level_util::AbortIfPanic;
    use crate::OwnedRepr;
    use crate::Zip;
    use rawpointer::PointerExt;
    use std::mem;
    use std::mem::MaybeUninit;
    impl<A, D> Array<A, D>
    where
        D: Dimension,
    {
        pub fn move_into_uninit<'a, AM>(self, new_array: AM)
        where
            AM: Into<ArrayViewMut<'a, MaybeUninit<A>, D>>,
            A: 'a,
        {
            unimplemented!()
        }
        fn move_into_impl(mut self, new_array: ArrayViewMut<MaybeUninit<A>, D>) {
            unimplemented!()
        }
        fn drop_unreachable_elements(mut self) -> OwnedRepr<A> {
            unimplemented!()
        }
        #[inline(never)]
        #[cold]
        fn drop_unreachable_elements_slow(mut self) -> OwnedRepr<A> {
            unimplemented!()
        }
        pub(crate) fn empty() -> Array<A, D> {
            unimplemented!()
        }
        #[cold]
        fn change_to_contig_append_layout(&mut self, growing_axis: Axis) {
            unimplemented!()
        }
        pub fn append(&mut self, axis: Axis, mut array: ArrayView<A, D>) -> Result<(), ShapeError>
        where
            A: Clone,
            D: RemoveAxis,
        {
            unimplemented!()
        }
    }
    pub(crate) unsafe fn drop_unreachable_raw<A, D>(
        mut self_: RawArrayViewMut<A, D>,
        data_ptr: *mut A,
        data_len: usize,
    ) where
        D: Dimension,
    {
        unimplemented!()
    }
    fn sort_axes_in_default_order<S, D>(a: &mut ArrayBase<S, D>)
    where
        S: RawData,
        D: Dimension,
    {
        unimplemented!()
    }
    fn sort_axes1_impl<D>(adim: &mut D, astrides: &mut D)
    where
        D: Dimension,
    {
        unimplemented!()
    }
    fn sort_axes_in_default_order_tandem<S, S2, D>(
        a: &mut ArrayBase<S, D>,
        b: &mut ArrayBase<S2, D>,
    ) where
        S: RawData,
        S2: RawData,
        D: Dimension,
    {
        unimplemented!()
    }
    fn sort_axes2_impl<D>(adim: &mut D, astrides: &mut D, bdim: &mut D, bstrides: &mut D)
    where
        D: Dimension,
    {
        unimplemented!()
    }
}
mod impl_special_element_types {
    use crate::imp_prelude::*;
    use crate::RawDataSubst;
    use std::mem::MaybeUninit;
    impl<A, S, D> ArrayBase<S, D>
    where
        S: RawDataSubst<A, Elem = MaybeUninit<A>>,
        D: Dimension,
    {
        pub unsafe fn assume_init(self) -> ArrayBase<<S as RawDataSubst<A>>::Output, D> {
            unimplemented!()
        }
    }
}
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
        unimplemented!()
    }
    #[inline]
    fn broadcast_assume<E>(&self, dim: E) -> ArrayView<'_, A, E>
    where
        E: Dimension,
    {
        unimplemented!()
    }
    fn try_remove_axis(self, axis: Axis) -> ArrayBase<S, D::Smaller> {
        unimplemented!()
    }
}
mod impl_ops {
    use num_complex::Complex;
    pub trait ScalarOperand: 'static + Clone {}
}
pub use crate::impl_ops::ScalarOperand;
mod impl_views {
    mod constructors {
        use crate::dimension;
        use crate::extension::nonnull::nonnull_debug_checked_from_ptr;
        use crate::imp_prelude::*;
        use crate::{is_aligned, StrideShape};
        use std::ptr::NonNull;
        impl<'a, A, D> ArrayView<'a, A, D>
        where
            D: Dimension,
        {
            pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *const A) -> Self
            where
                Sh: Into<StrideShape<D>>,
            {
                unimplemented!()
            }
        }
        impl<'a, A, D> ArrayViewMut<'a, A, D>
        where
            D: Dimension,
        {
            pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *mut A) -> Self
            where
                Sh: Into<StrideShape<D>>,
            {
                unimplemented!()
            }
        }
        impl<'a, A, D> ArrayView<'a, A, D>
        where
            D: Dimension,
        {
            #[inline(always)]
            pub(crate) unsafe fn new(ptr: NonNull<A>, dim: D, strides: D) -> Self {
                unimplemented!()
            }
            #[inline]
            pub(crate) unsafe fn new_(ptr: *const A, dim: D, strides: D) -> Self {
                unimplemented!()
            }
        }
        impl<'a, A, D> ArrayViewMut<'a, A, D>
        where
            D: Dimension,
        {
            #[inline(always)]
            pub(crate) unsafe fn new(ptr: NonNull<A>, dim: D, strides: D) -> Self {
                unimplemented!()
            }
            #[inline(always)]
            pub(crate) unsafe fn new_(ptr: *mut A, dim: D, strides: D) -> Self {
                unimplemented!()
            }
        }
    }
    mod conversions {
        use crate::imp_prelude::*;
        use crate::{Baseiter, ElementsBase, ElementsBaseMut, Iter, IterMut};
        use alloc::slice;
        impl<'a, A, D> ArrayView<'a, A, D>
        where
            D: Dimension,
        {
            pub fn to_slice(&self) -> Option<&'a [A]> {
                unimplemented!()
            }
        }
        impl<'a, A, D> ArrayView<'a, A, D>
        where
            D: Dimension,
        {
            #[inline]
            pub(crate) fn into_base_iter(self) -> Baseiter<A, D> {
                unimplemented!()
            }
            #[inline]
            pub(crate) fn into_elements_base(self) -> ElementsBase<'a, A, D> {
                unimplemented!()
            }
            pub(crate) fn into_iter_(self) -> Iter<'a, A, D> {
                unimplemented!()
            }
        }
        impl<'a, A, D> ArrayViewMut<'a, A, D>
        where
            D: Dimension,
        {
            pub(crate) fn into_raw_view_mut(self) -> RawArrayViewMut<A, D> {
                unimplemented!()
            }
            #[inline]
            pub(crate) fn into_base_iter(self) -> Baseiter<A, D> {
                unimplemented!()
            }
            #[inline]
            pub(crate) fn into_elements_base(self) -> ElementsBaseMut<'a, A, D> {
                unimplemented!()
            }
            pub(crate) fn try_into_slice(self) -> Result<&'a mut [A], Self> {
                unimplemented!()
            }
            pub(crate) fn into_iter_(self) -> IterMut<'a, A, D> {
                unimplemented!()
            }
        }
    }
    mod indexing {
        pub trait IndexLonger<I> {
            type Output;
            fn index(self, index: I) -> Self::Output;
            fn get(self, index: I) -> Option<Self::Output>;
            unsafe fn uget(self, index: I) -> Self::Output;
        }
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
        #[inline]
        pub(crate) unsafe fn new(ptr: NonNull<A>, dim: D, strides: D) -> Self {
            unimplemented!()
        }
        unsafe fn new_(ptr: *const A, dim: D, strides: D) -> Self {
            unimplemented!()
        }
        pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *const A) -> Self
        where
            Sh: Into<StrideShape<D>>,
        {
            unimplemented!()
        }
        #[inline]
        pub unsafe fn deref_into_view<'a>(self) -> ArrayView<'a, A, D> {
            unimplemented!()
        }
    }
    impl<A, D> RawArrayViewMut<A, D>
    where
        D: Dimension,
    {
        #[inline]
        pub(crate) unsafe fn new(ptr: NonNull<A>, dim: D, strides: D) -> Self {
            unimplemented!()
        }
        unsafe fn new_(ptr: *mut A, dim: D, strides: D) -> Self {
            unimplemented!()
        }
        pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *mut A) -> Self
        where
            Sh: Into<StrideShape<D>>,
        {
            unimplemented!()
        }
        #[inline]
        pub unsafe fn deref_into_view_mut<'a>(self) -> ArrayViewMut<'a, A, D> {
            unimplemented!()
        }
        pub fn cast<B>(self) -> RawArrayViewMut<B, D> {
            unimplemented!()
        }
    }
}
pub(crate) fn is_aligned<T>(ptr: *const T) -> bool {
    unimplemented!()
}
