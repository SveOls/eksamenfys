use std::{
    error::Error,
    io::{BufRead, BufReader},
};

use plotters::prelude::*;

pub fn _data_manage(path: &str) -> Result<Vec<[f64; 4]>, Box<dyn Error>> {
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

/// x axis: sweeps
///
/// y_left: checking (1 = energy, 2 = end to end, 3 = rog)
///
/// y_right: temperature
pub fn doubledraw(
    data: &[[f64; 4]],
    checking: usize,
    save: &str,
    title: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(save, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let data = &data[data.len()/10..9*data.len()/10];

    let maxt = data
        .iter()
        .enumerate()
        .map(|(_, x)| x[0])
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let mint = data
        .iter()
        .enumerate()
        .map(|(_, x)| x[0])
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();

    let minsw = 0;
    let maxsw =
        data.iter().filter(|x| x[0] > mint).count() + data.iter().filter(|x| x[0] == maxt).count();


    let avg_high = data.iter().filter(|x| x[0] > (maxt - mint)*0.95 + mint).map(|x| x[checking]).enumerate().fold(0.0, |acc, x| (x.0 as f64 * acc + x.1) / (x.0+1) as f64);
    let avg_low  = data.iter().filter(|x| x[0] < (maxt - mint)*0.05 + mint).map(|x| x[checking]).enumerate().fold(0.0, |acc, x| (x.0 as f64 * acc + x.1) / (x.0+1) as f64);

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .right_y_label_area_size(40)
        .caption(title.unwrap_or_default(), ("sans-serif", 50.0).into_font())
        .build_cartesian_2d(minsw..maxsw, (avg_low - (avg_high - avg_low) / 10.0)..(avg_high + (avg_high - avg_low) / 10.0))?
        .set_secondary_coord(minsw..maxsw, (mint - (maxt - mint) / 10.0)..(maxt + (maxt - mint) / 10.0));

    chart.configure_secondary_axes().draw()?;
    chart.configure_mesh().max_light_lines(1).draw()?;

    chart.draw_series(LineSeries::new(
        data.iter()
            .enumerate()
            .map(|(i, y)| (i, y[checking])),
        &BLUE,
    ))?;
    chart.draw_secondary_series(LineSeries::new(data.iter().map(|y| y[0]).enumerate(), &RED))?;

    Ok(())
}

pub fn singledraw(
    data: &Vec<[f64; 4]>,
    checking: usize,
    save: &str,
    title: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(save, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let minv = data
        .iter()
        .map(|x| x[checking])
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let maxv = data
        .iter()
        .map(|x| x[checking])
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let mint = 0;
    let maxt = data.len();

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .caption(title.unwrap_or_default(), ("sans-serif", 50.0).into_font())
        .build_cartesian_2d(mint..maxt, minv..maxv)?;

    chart.configure_mesh().max_light_lines(1).draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|y| y[checking]).enumerate(),
        &BLUE,
    ))?;

    Ok(())
}

pub fn phasedraw(
    data: &[[f64; 4]],
    checking: usize,
    save: &str,
    title: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // println!("{data:?}");

    let data = &data[data.len()/10..9*data.len()/10];

    let root = BitMapBackend::new(save, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

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
        .caption(title.unwrap_or_default(), ("sans-serif", 50.0).into_font())
        .build_cartesian_2d(mint..maxt, minv..maxv)?;

    chart.configure_mesh().max_light_lines(1).draw()?;

    chart.draw_series(LineSeries::new(
        data.iter()
            .map(|y| (y[0], y[checking]))
            .filter(|x| x.0 != 20.0 && x.0 != 1.0),
        &BLUE,
    ))?;
    Ok(())
}
