use std::{
    error::Error,
    io::{BufRead, BufReader},
};

use plotters::prelude::*;

pub fn data_manage(path: &str) -> Result<Vec<[f64; 4]>, Box<dyn Error>> {
    let mut tempdata = [f64::NAN; 4];
    let mut data1 = Vec::new();
    for i in BufReader::new(std::fs::File::open(format!("{path}.txt"))?)
        .lines()
        .filter_map(|x| x.ok())
    {
        match i.as_str().split_once(':') {
            Some(("START", _)) => tempdata = [f64::NAN; 4],
            Some(("energy", a)) => tempdata[0] = a.trim().parse()?,
            Some(("end_to_end", a)) => tempdata[1] = a.trim().parse()?,
            Some(("temperature", a)) => tempdata[2] = a.trim().parse()?,
            Some(("rog", a)) => tempdata[3] = a.trim().parse()?,
            Some(("END", _)) => data1.push(tempdata),
            _ => {}
        }
    }
    let mut data2 = Vec::new();
    for i in 5..data1.len() - 5 {
        data2.push([
            data1[i][2],
            (data1[i][0]
                + data1[i + 1][0]
                + 0.5 * data1[i + 2][0]
                + 0.25 * data1[i + 3][0]
                + 0.125 * data1[i + 4][0]
                + 0.0625 * data1[i + 5][0]
                + data1[i - 1][0]
                + 0.5 * data1[i - 2][0]
                + 0.25 * data1[i - 3][0]
                + 0.125 * data1[i - 4][0]
                + 0.0625 * data1[i - 5][0])
                / (1.0 + 2.0 + 1.0 + 0.5 + 0.25 + 0.125),
            (data1[i][3]
                + data1[i + 1][3]
                + 0.5 * data1[i + 2][3]
                + 0.25 * data1[i + 3][3]
                + 0.125 * data1[i + 4][3]
                + 0.0625 * data1[i + 5][3]
                + data1[i - 1][3]
                + 0.5 * data1[i - 2][3]
                + 0.25 * data1[i - 3][3]
                + 0.125 * data1[i - 4][3]
                + 0.0625 * data1[i - 5][3])
                / (1.0 + 2.0 + 1.0 + 0.5 + 0.25 + 0.125),
            data1[i][1],
        ])
    }
    Ok(data2)
}

pub fn draw2(data: Vec<[[f64; 4]; 2]>) -> Result<(), Box<dyn Error>> {
    println!("{data:?}");

    let root = BitMapBackend::new("out.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let checking = 1;
    // let min = data
    //     .iter()
    //     .map(|x| x[checking])
    //     .min_by(|x, y| x.partial_cmp(y).unwrap())
    //     .unwrap();
    // let max = data
    //     .iter()
    //     .map(|x| x[checking])
    //     .max_by(|x, y| x.partial_cmp(y).unwrap())
    //     .unwrap();

    // let mut chart = ChartBuilder::on(&root)
    //     .x_label_area_size(40)
    //     .y_label_area_size(40)
    //     .caption("M", ("sans-serif", 50.0).into_font())
    //     .build_cartesian_2d(0..data.len(), min..max)?;

    // chart.configure_mesh().draw()?;

    // chart.draw_series(LineSeries::new(
    //     data.iter().enumerate().map(|(x, y)| (x, y[checking])),
    //     &BLUE,
    // ))?;

    let minv = data
        .iter()
        .map(|x| x[0][checking].min(x[1][checking]))
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let maxv = data
        .iter()
        .map(|x| x[0][checking].max(x[1][checking]))
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let mint = data
        .iter()
        .map(|x| x[0][0].min(x[1][0]))
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let maxt = data
        .iter()
        .map(|x| x[0][0].max(x[1][0]))
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();

    // println!("{:?}", data.iter().enumerate().max_by(|x, y| x.1[checking].partial_cmp(&y.1[checking]).unwrap()));
    // println!("{:?}", data.iter().enumerate().min_by(|x, y| x.1[checking].partial_cmp(&y.1[checking]).unwrap()));

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .caption("M", ("sans-serif", 50.0).into_font())
        .build_cartesian_2d(mint..maxt, minv..maxv)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|y| (y[0][0], y[0][checking])),
        &BLUE,
    ))?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|y| (y[1][0], y[1][checking])),
        &RED,
    ))?;


    Ok(())
}
pub fn draw(data: Vec<[f64; 4]>) -> Result<(), Box<dyn Error>> {
    println!("{data:?}");

    let root = BitMapBackend::new("out1.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let checking = 1;

    let minv = data
        .iter()
        .filter(|x| x[0] != 20.0 && x[0] != 1.0)
        .map(|x| x[checking])
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let maxv = data
        .iter()
        .filter(|x| x[0] != 20.0 && x[0] != 1.0)
        .map(|x| x[checking])
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let mint = data
        .iter()
        .map(|x| x[0])
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let maxt = data
        .iter()
        .map(|x| x[0])
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();

    // println!("{:?}", data.iter().enumerate().max_by(|x, y| x.1[checking].partial_cmp(&y.1[checking]).unwrap()));
    // println!("{:?}", data.iter().enumerate().min_by(|x, y| x.1[checking].partial_cmp(&y.1[checking]).unwrap()));

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .caption("E", ("sans-serif", 50.0).into_font())
        .build_cartesian_2d(mint..maxt, minv..maxv)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|y| (y[0], y[checking])).filter(|x| x.0 != 20.0 && x.0 != 1.0),
        &BLUE,
    ))?;


    let root = BitMapBackend::new("out2.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let checking = 2;

    let minv = data
        .iter()
        .filter(|x| x[0] != 20.0 && x[0] != 1.0)
        .map(|x| x[checking])
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let maxv = data
        .iter()
        .filter(|x| x[0] != 20.0 && x[0] != 1.0)
        .map(|x| x[checking])
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let mint = data
        .iter()
        .map(|x| x[0])
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let maxt = data
        .iter()
        .map(|x| x[0])
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();

    // println!("{:?}", data.iter().enumerate().max_by(|x, y| x.1[checking].partial_cmp(&y.1[checking]).unwrap()));
    // println!("{:?}", data.iter().enumerate().min_by(|x, y| x.1[checking].partial_cmp(&y.1[checking]).unwrap()));

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .caption("e2e", ("sans-serif", 50.0).into_font())
        .build_cartesian_2d(mint..maxt, minv..maxv)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|y| (y[0], y[checking])).filter(|x| x.0 != 20.0 && x.0 != 1.0),
        &BLUE,
    ))?;

    let root = BitMapBackend::new("out3.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let checking = 3;

    let minv = data
        .iter()
        .filter(|x| x[0] != 20.0 && x[0] != 1.0)
        .map(|x| x[checking])
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let maxv = data
        .iter()
        .filter(|x| x[0] != 20.0 && x[0] != 1.0)
        .map(|x| x[checking])
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let mint = data
        .iter()
        .map(|x| x[0])
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let maxt = data
        .iter()
        .map(|x| x[0])
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();

    // println!("{:?}", data.iter().enumerate().max_by(|x, y| x.1[checking].partial_cmp(&y.1[checking]).unwrap()));
    // println!("{:?}", data.iter().enumerate().min_by(|x, y| x.1[checking].partial_cmp(&y.1[checking]).unwrap()));

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .caption("rog", ("sans-serif", 50.0).into_font())
        .build_cartesian_2d(mint..maxt, minv..maxv)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|y| (y[0], y[checking])).filter(|x| x.0 != 20.0 && x.0 != 1.0),
        &BLUE,
    ))?;

    Ok(())
}
