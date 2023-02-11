use std::mem;
use std::mem::size_of;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use std::collections::HashMap;
type TractResult<T> = Result<T, ()>;
#[derive(Copy, Clone)]
enum BinOp {
    Min,
    Max,
}
#[derive(Clone, Copy)]
enum OutputStoreSpec {
    View {
        m_axis: usize,
    },
    Strides {
        col_byte_stride: isize,
        mr: usize,
        nr: usize,
        m: usize,
        n: usize,
    },
}
#[derive(Clone)]
enum ProtoFusedSpec {
    BinScalar(AttrOrInput, BinOp),
    BinPerRow(AttrOrInput, BinOp),
    BinPerCol(AttrOrInput, BinOp),
    AddRowColProducts(AttrOrInput, AttrOrInput),
    AddUnicast(OutputStoreSpec, AttrOrInput),
    Store,
}
#[derive(Clone)]
enum AttrOrInput {
    Attr(Arc<()>),
    Input(usize),
}
/*
fn dump_pfs(pfs: &ProtoFusedSpec) {
    let ptr = pfs as *const ProtoFusedSpec as *const u8;
    for i in 0..std::mem::size_of::<ProtoFusedSpec>() {
        let v = unsafe { *ptr.add(i) };
        if v == 0 {
            print!("__ ");
        } else {
            print!("{:02x} ", v);
        }
        if i % 8 == 7 {
            print!("| ");
        }
    }
    println!("");
}
*/
fn crasher_monterey() {
    let mut stuff = vec!(vec!(1));
    for i in 0..10000 {
	let len = (stuff[i].len() * 134775813) % 4096;
        stuff.push((0..).take(len).collect());
    }
    std::mem::drop(stuff);
    vec!((Box::new(()), vec![ProtoFusedSpec::Store])).as_slice().to_owned();
}
#[test]
fn t1() {
    crasher_monterey()
}
