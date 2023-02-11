#![allow(dead_code)]
#[derive(Copy, Clone)]
enum BinOp {
    Min,
}
#[derive(Clone, Copy)]
enum OutputStoreSpec {
    View(usize),
    Strides([isize; 5])
}
#[derive(Clone)]
enum AttrOrInput {
    Attr(Box<()>),
    Input(usize),
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
fn main() {
    let mut stuff = vec!(vec!(1));
    for i in 0..50000 {
	let len = (stuff[i].len() * 134775813) % 4096;
        stuff.push((1234123414u32..).take(len).collect());
    }
    std::mem::drop(stuff);
    let _ = vec!((Box::new(()), vec![ProtoFusedSpec::Store])).as_slice().to_owned();
}
