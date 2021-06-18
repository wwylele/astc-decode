use astc_decode::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::*;
use rand_pcg::*;

fn benchmark_fp<const W: usize, const H: usize>(c: &mut Criterion) {
    let mut rng = Pcg64::seed_from_u64(42);
    let footprint = Footprint::new(W as u32, H as u32);

    const INPUT_SET_SIZE: usize = 10000;

    // First prepare a set of valid input
    let mut inputs = vec![];
    while inputs.len() < INPUT_SET_SIZE {
        let block = rng.sample(distributions::Standard);
        if astc_decode_block(&block, footprint, |_, _, _| ()) {
            inputs.push(block);
        }
    }

    // Then benchmark against valid input
    c.bench_function(&format!("astc {}x{}", W, H), |b| {
        b.iter(|| {
            let mut sink = [[[0; 4]; H]; W];
            let block = inputs[rng.sample(distributions::Uniform::new(0, INPUT_SET_SIZE))];
            astc_decode_block(&black_box(block), footprint, |x, y, v| {
                sink[x as usize][y as usize] = v
            });
            sink
        })
    });
}

fn benchmark(c: &mut Criterion) {
    benchmark_fp::<4, 4>(c);
    benchmark_fp::<10, 5>(c);
    benchmark_fp::<12, 12>(c);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
