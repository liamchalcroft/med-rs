//! Preprocess an image with a fused pipeline.
//!
//! Run: `cargo run --release --example preprocess -- input.nii.gz output.nii.gz`

use medrs::transforms::{Interpolation, Orientation};
use medrs::Pipeline;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(input), Some(output)) = (args.next(), args.next()) else {
        eprintln!("usage: preprocess <input> <output>");
        std::process::exit(2);
    };

    let image = medrs::load(&input)?;
    println!("{image:?} orientation {}", image.orientation());

    let pipeline = Pipeline::new()
        .reorient(Orientation::RAS)
        .resample_to_spacing([1.0, 1.0, 1.0], Interpolation::Trilinear)
        .crop_or_pad([160, 192, 160], 0.0)
        .z_normalize_nonzero();

    let start = Instant::now();
    let processed = pipeline.apply(&image)?;
    println!("pipeline: {:.1} ms", start.elapsed().as_secs_f64() * 1e3);

    medrs::save(&processed, &output)?;
    println!("wrote {output}: {processed:?}");
    Ok(())
}
