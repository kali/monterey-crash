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
pub use crate::dimension::{Axis, AxisDescription, Dimension, IntoDimension, RemoveAxis};
pub use crate::dimension::{DimAdd, DimMax};
pub use crate::indexes::{indices, indices_of};
pub use crate::shape_builder::{Shape, ShapeArg, ShapeBuilder, StrideShape};
use alloc::sync::Arc;
use std::marker::PhantomData;
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
    pub fn Ix1(i0: Ix) -> Ix1 {
        Dim::new([i0])
    }
    pub type Ix0 = Dim<[Ix; 0]>;
    pub type Ix1 = Dim<[Ix; 1]>;
    pub type Ix2 = Dim<[Ix; 2]>;
    pub type Ix3 = Dim<[Ix; 3]>;
    pub type Ix4 = Dim<[Ix; 4]>;
    pub type Ix5 = Dim<[Ix; 5]>;
    pub type Ix6 = Dim<[Ix; 6]>;
    pub type IxDyn = Dim<IxDynImpl>;
    pub type ArrayView1<'a, A> = ArrayView<'a, A, Ix1>;
    pub type ArrayViewMut1<'a, A> = ArrayViewMut<'a, A, Ix1>;
}
#[macro_use]
mod itertools {
    use std::iter;
    pub(crate) fn enumerate<I>(iterable: I) -> iter::Enumerate<I::IntoIter>
    where
        I: IntoIterator,
    {
        unimplemented!()
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
mod data_repr {
    use crate::extension::nonnull;
    use alloc::borrow::ToOwned;
    use alloc::slice;
    use alloc::vec::Vec;
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
        pub(crate) fn as_ptr(&self) -> *const A {
            self.ptr.as_ptr()
        }
        pub(crate) fn as_nonnull_mut(&mut self) -> NonNull<A> {
            self.ptr
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
    use alloc::vec::Vec;
    use std::mem::MaybeUninit;
    use std::mem::{self, size_of};
    use std::ptr::NonNull;
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait RawData: Sized {
        type Elem;
        #[deprecated(note = "Unused", since = "0.15.2")]
        fn _data_slice(&self) -> Option<&[Self::Elem]>;
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
    unsafe impl<A> RawData for OwnedArcRepr<A> {
        type Elem = A;
        fn _data_slice(&self) -> Option<&[A]> {
            unimplemented!()
        }
        private_impl! {}
    }
    unsafe impl<A> RawData for OwnedRepr<A> {
        type Elem = A;
        fn _data_slice(&self) -> Option<&[A]> {
            unimplemented!()
        }
        private_impl! {}
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
                new_ptr = NonNull::new(new_ptr.as_ptr().offset(our_off)).unwrap();
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
        private_impl! {}
    }
    unsafe impl<'a, A> RawData for ViewRepr<&'a mut A> {
        type Elem = A;
        #[inline]
        fn _data_slice(&self) -> Option<&[A]> {
            unimplemented!()
        }
        private_impl! {}
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
pub use crate::iterators::iter;
mod error {
    #[derive(Clone)]
    pub struct ShapeError {
        repr: ErrorKind,
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
}
mod extension {
    pub(crate) mod nonnull {
        use alloc::vec::Vec;
        use std::ptr::NonNull;
        pub(crate) fn nonnull_from_vec_data<T>(v: &mut Vec<T>) -> NonNull<T> {
            unsafe { NonNull::new_unchecked(v.as_mut_ptr()) }
        }
    }
}
mod indexes {
    use super::Dimension;
    use crate::dimension::IntoDimension;
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
}
mod iterators {
    pub mod iter {
        pub use crate::indexes::{Indices, IndicesIter};
    }
    use super::{Dimension, Ix, Ixs};
    use alloc::vec::Vec;
    use std::ptr;
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait TrustedIterator {}
    use crate::iter::IndicesIter;
    unsafe impl<D> TrustedIterator for IndicesIter<D> where D: Dimension {}
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
mod order {
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[non_exhaustive]
    pub enum Order {
        RowMajor,
        ColumnMajor,
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
mod dimension {
    pub use self::axes::{Axes, AxisDescription};
    pub use self::axis::Axis;
    pub use self::broadcast::DimMax;
    pub use self::conversion::IntoDimension;
    pub use self::dim::*;
    pub use self::dimension_trait::Dimension;
    pub use self::dynindeximpl::IxDynImpl;
    pub use self::ops::DimAdd;
    pub use self::remove_axis::RemoveAxis;
    use crate::error::{from_kind, ErrorKind, ShapeError};
    mod axes {
        use crate::{Axis, Dimension, Ix, Ixs};
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
        use super::conversion::Convert;
        use super::ops::DimAdd;
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
                unimplemented!()
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
                unimplemented!()
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
        use crate::{
            Dim, Dimension, IntoDimension, Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, IxDynImpl,
        };
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
}
mod imp_prelude {
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
#[derive(Copy, Clone)]
pub struct RawViewRepr<A> {
    ptr: PhantomData<A>,
}
#[derive(Copy, Clone)]
pub struct ViewRepr<A> {
    life: PhantomData<A>,
}
pub enum CowRepr<'a, A> {
    View(ViewRepr<&'a A>),
    Owned(OwnedRepr<A>),
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
        pub(crate) unsafe fn from_data_ptr(data: S, ptr: NonNull<A>) -> Self {
            let array = ArrayBase {
                data,
                ptr,
                dim: Ix1(0),
                strides: Ix1(1),
            };
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
    use crate::imp_prelude::*;
    use crate::indices;
    use crate::iterators::to_vec_mapped;
    use crate::StrideShape;
    use alloc::vec::Vec;
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
            let ptr = std::ptr::NonNull::new(
                v.as_mut_ptr()
                    .add(offset_from_low_addr_ptr_to_logical_ptr(&dim, &strides)),
            )
            .unwrap();
            ArrayBase::from_data_ptr(DataOwned::new(v), ptr).with_strides_dim(strides, dim)
        }
    }
}
