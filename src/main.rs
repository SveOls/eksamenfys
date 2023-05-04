#![feature(file_create_new)]
mod plotting;

use std::{error::Error, fmt::Display, fs::File, io::Write, sync::Mutex, sync::Arc};

use rand::random;

/// turns f64 and f32 into bits, increments or decrements the significand,
/// and turns them back into f64 and f32, adjusts exponent by subtracting 1.0, and
/// prints in scienfific notation.
fn tester() {
    println!("{:+e}", f64::from_bits(1.0f64.to_bits() + 1) - 1.0);
    println!("{:+e}", f64::from_bits(1.0f64.to_bits() - 1) - 1.0);
    println!("{:+e}", f32::from_bits(1.0f32.to_bits() + 1) - 1.0);
    println!("{:+e}", f32::from_bits(1.0f32.to_bits() - 1) - 1.0);
}

fn main() -> Result<(), Box<dyn Error>> {
    tester();

    let range = 20;
    let temperature = 20.0;
    let mintemp = 0.1;
    // margin
    let margin = 5f64;
    let n = 100;
    let sweeps = 2000;
    let sample_per_thread = 100;
    let threads = 5;
    const DIM: usize = 2;

    // function that calculates temperature - exponential decay from
    let temp = move |inp: f64| -> f64 {
        (margin * temperature * (((mintemp / margin.powi(2))/temperature).ln() * inp).exp()).clamp(mintemp, temperature)
    };

    let data = Arc::new(Mutex::new(vec![[0.0; 4]; sweeps]));
    let mut handles = Vec::new();
    for _ in 0..threads {
        let data = Arc::clone(&data);
        let handle = std::thread::spawn(move || {
            let mut ret = vec![[0.0; 4]; sweeps];
            for w in 0..sample_per_thread {
                print!("\r{:>6.1}%\r", 100.0 * w as f64 / sample_per_thread as f64);
                std::io::stdout().flush().unwrap();
                let structure: Chain2d<DIM> = Chain2d::new_linear(Some(range), n, temperature);
                let new = structure.eat(sweeps, Box::new(temp), false, false).unwrap();
                let tempe = |x: f64, y: f64| -> f64 { (w as f64 * y + x) / (w as f64 + 1.0) };
                ret.iter_mut().enumerate().for_each(|(i, x)| {
                    x[0] = new[i][0];
                    x[1] = tempe(0.5 * new[i][1] + 0.25 * i.checked_sub(1).map(|t| new.get(t).map(|l| l[1])).flatten().unwrap_or(new[i][1]) + 0.25 * i.checked_sub(1).map(|t| new.get(t).map(|l| l[1])).flatten().unwrap_or(new[i][1]), x[1]);
                    x[2] = tempe(0.5 * new[i][2] + 0.25 * i.checked_sub(1).map(|t| new.get(t).map(|l| l[2])).flatten().unwrap_or(new[i][2]) + 0.25 * i.checked_sub(1).map(|t| new.get(t).map(|l| l[2])).flatten().unwrap_or(new[i][2]), x[2]);
                    x[3] = tempe(0.5 * new[i][3] + 0.25 * i.checked_sub(1).map(|t| new.get(t).map(|l| l[3])).flatten().unwrap_or(new[i][3]) + 0.25 * i.checked_sub(1).map(|t| new.get(t).map(|l| l[3])).flatten().unwrap_or(new[i][3]), x[3]);
                    // x[1] = tempe(new[i][1], x[1]);
                    // x[2] = tempe(new[i][2], x[2]);
                    // x[3] = tempe(new[i][3], x[3]);
                });
                // panic!();
            }
            let mut data = data.lock().unwrap();
            data.iter_mut().zip(ret.into_iter()).for_each(|(x, y)| {
                x[0] = y[0];
                x[1] += y[1] / threads as f64;
                x[2] += y[2] / threads as f64;
                x[3] += y[3] / threads as f64;
            });
        });
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap();
    }
    let data = Arc::<Mutex<Vec<[f64; 4]>>>::try_unwrap(data).unwrap().into_inner()?;
    plotting::draw(data)?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Monomer<const N: usize> {
    m_type: usize,
    position: [i64; N],
    neighbors: [[Option<usize>; 2]; N],
}

impl<const N: usize> Monomer<N> {
    pub fn m_type(&self) -> usize {
        self.m_type
    }
    pub fn neighbors(&self) -> &[[Option<usize>; 2]; N] {
        &self.neighbors
    }
    pub fn position(&self) -> [i64; N] {
        self.position
    }
    pub fn move_to(&mut self, pos: [i64; N]) {
        self.position = pos;
    }
    pub fn new(m_type: usize, position: [i64; N]) -> Self {
        Self {
            m_type,
            position,
            neighbors: [[None; 2]; N],
        }
    }
}

#[derive(Debug, Default)]
struct Chain2d<const N: usize> {
    mon: Vec<Monomer<N>>,
    interaction_e: Vec<f64>,
    last_energy: f64,
    last_distance: f64,
    temperature: f64,
    logger: Option<File>,
    logvec: Vec<[f64; 4]>,
    itercount: usize,
    sweep_size: usize,
}

impl<const N: usize> Iterator for Chain2d<N> {
    // changed index
    type Item = Option<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        self.itercount += 1;
        // loops until a monomer that can move is found. Will loop forever if the polymer is stuck.

        let test_index = random::<usize>() % self.mon.len();
        if self.valid_positions(test_index).len() == 0 {
            return Some(None);
        }
        let pos = self.valid_positions(test_index)
            [random::<usize>() % self.valid_positions(test_index).len()];

        // saved old position of monomer
        let old_pos = self.mon[test_index].position();
        // monomermove
        self.mon[test_index].move_to(pos);
        // jehova's witness activities
        self.update_neighbors(Some(test_index));
        // saved to variable to avoid double calculations
        let energy = self.energy();
        // if exp(- delta_E / T) > random number [0, 1]
        if ((self.last_energy - energy) / self.temperature)
            .exp()
            .min(1.0)
            > random::<f64>()
        {
            // saving variables
            self.last_energy = energy;
            self.last_distance = self.end_to_end();
            // some(some(index)) to differentiate between a termination and a non-move
            Some(Some(test_index))
        } else {
            // restoration of old data
            self.mon[test_index].move_to(old_pos);
            self.update_neighbors(Some(test_index));
            Some(None)
        }
    }
}

impl<const N: usize> Display for Chain2d<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if N != 2 {
            return Ok(())
        }
        let max = self.mon.iter().fold([i64::MAX, i64::MIN], |acc, x| {
            [acc[0].min(x.position()[0]), acc[1].max(x.position()[0])]
        });
        let may = self.mon.iter().fold([i64::MAX, i64::MIN], |acc, x| {
            [acc[0].min(x.position()[1]), acc[1].max(x.position()[1])]
        });
        let maymin;
        if may[0] < 1 {
            maymin = 1;
        } else {
            maymin = 0;
        }
        let maxmin;
        if max[0] < 1 {
            maxmin = 1;
        } else {
            maxmin = 0;
        }
        let ywidth = ((may[1].abs().max(1).ilog10())
            .max((may[0] - 1).abs().max(1).ilog10() + maymin)
            + 2) as usize;
        let xwidth = ((max[1].abs().max(1).ilog10())
            .max((max[0] - 1).abs().max(1).ilog10() + maxmin)
            + 2) as usize;
        for y in (may[0] - 1..=may[1] + 1).rev() {
            write!(f, "{:1$} |", y, ywidth)?;
            for x in max[0] - 1..=max[1] + 1 {
                if let Some(a) = self.mon.iter().position(|z| &z.position()[0..2] == &[x, y]) {
                    if a == 0 {
                        if self.mon[a + 1].position()[0] == self.mon[a].position()[0] + 1 {
                            write!(f, "{:2$}{:━<3$}", "", "o", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        } else if self.mon[a + 1].position()[0] == self.mon[a].position()[0] - 1 {
                            write!(f, "{:━<2$}{:3$}", "", "o", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        } else {
                            write!(f, "{:^1$}", "o", xwidth)?;
                        }
                    } else if a == self.mon.len() - 1 {
                        if self.mon[a - 1].position()[0] == self.mon[a].position()[0] + 1 {
                            write!(f, "{:2$}{:━<3$}", "", "o", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        } else if self.mon[a - 1].position()[0] == self.mon[a].position()[0] - 1 {
                            write!(f, "{:━<2$}{:3$}", "", "o", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        } else {
                            write!(f, "{:^1$}", "o", xwidth)?;
                        }
                    } else if self.mon[a - 1].position()[0] == self.mon[a + 1].position()[0] {
                        write!(f, "{:^1$}", "┃", xwidth)?;
                    } else if self.mon[a - 1].position()[1] == self.mon[a + 1].position()[1] {
                        write!(
                            f,
                            "{:━^2$}{:━<3$}",
                            "",
                            "━",
                            (xwidth - 1) / 2,
                            xwidth / 2 + 1
                        )?;
                    } else if self.mon[a - 1].position()[1] == self.mon[a].position()[1] + 1 {
                        if self.mon[a + 1].position()[0] == self.mon[a].position()[0] + 1 {
                            write!(f, "{:2$}{:━<3$}", "", "┗", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        } else {
                            write!(
                                f,
                                "{:━^2$}{:<3$}",
                                "",
                                "┛",
                                (xwidth - 1) / 2,
                                xwidth / 2 + 1
                            )?;
                        }
                    } else if self.mon[a + 1].position()[1] == self.mon[a].position()[1] + 1 {
                        if self.mon[a - 1].position()[0] == self.mon[a].position()[0] + 1 {
                            write!(f, "{:2$}{:━<3$}", "", "┗", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        } else {
                            write!(
                                f,
                                "{:━^2$}{:<3$}",
                                "",
                                "┛",
                                (xwidth - 1) / 2,
                                xwidth / 2 + 1
                            )?;
                        }
                    } else if self.mon[a - 1].position()[1] == self.mon[a].position()[1] - 1 {
                        if self.mon[a + 1].position()[0] == self.mon[a].position()[0] - 1 {
                            write!(
                                f,
                                "{:━^2$}{:<3$}",
                                "",
                                "┓",
                                (xwidth - 1) / 2,
                                xwidth / 2 + 1
                            )?;
                        } else {
                            write!(f, "{:2$}{:━<3$}", "", "┏", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        }
                    } else if self.mon[a + 1].position()[1] == self.mon[a].position()[1] - 1 {
                        if self.mon[a - 1].position()[0] == self.mon[a].position()[0] - 1 {
                            write!(
                                f,
                                "{:━^2$}{:<3$}",
                                "",
                                "┓",
                                (xwidth - 1) / 2,
                                xwidth / 2 + 1
                            )?;
                        } else {
                            write!(f, "{:2$}{:━<3$}", "", "┏", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        }
                    } else {
                        write!(f, "{:^1$}", "x", xwidth)?;
                    }
                } else {
                    write!(f, "{:^1$}", ".", xwidth)?;
                }
            }
            writeln!(f)?;
        }
        writeln!(
            f,
            "{0:-<1$}-|{0:-<2$}",
            "",
            ywidth,
            xwidth * (max[1] - max[0] + 3) as usize
        )?;
        write!(f, "{0:<1$} |", "", ywidth)?;
        for x in max[0] - 1..=max[1] + 1 {
            write!(f, "{:1$}", x, xwidth)?;
        }
        writeln!(f)
    }
}

impl<const N: usize> Chain2d<N> {

    fn sweeps(&self) -> usize {
        self.itercount / self.sweep_size
    }
    fn sweep(&mut self) {
        self.nth(self.sweep_size - 1);
    }
    /// distance between ends of polymer
    fn end_to_end(&self) -> f64 {
        let fpos = self.mon.first().map(|x| x.position()).unwrap();
        let lpos = self.mon.last().map(|x| x.position()).unwrap();
        fpos.iter().zip(lpos.iter()).fold(0.0, |acc, z| acc + (*z.0 - *z.1).pow(2) as f64).sqrt()
    }
    fn rog(&self) -> f64 {
        let center =
            self.mon
                .iter()
                .map(|x| x.position())
                .fold([0.0f64; N], |acc, x| {
                    let mut ret = [0.0; N];
                    for i in 0..N {
                        ret[i] = acc[i] + (x[i] as f64 / self.mon.len() as f64)
                    }
                    ret
            });
        let mut ret = 0.0;
        for i in self.mon.iter().map(|x| x.position()) {
            ret += center.iter().zip(i.iter()).fold(0.0, |acc, x| acc + (x.0 - *x.1 as f64).powi(2))
        }
        // println!("{}, {:?}, \n{}", ret, center, self);
        ret
    }
    /// energy
    /// iterates through every monomer, and turns them into an iterator over connections.
    /// as each connection appears twice, the final product is divided by 2.
    /// self.interaction_e is the 20x20 matrix of interaction energies.
    fn energy(&self) -> f64 {
        self.mon
            .iter()
            .enumerate()
            .flat_map(|x| {
                x.1.neighbors()
                    .into_iter()
                    .flat_map(|x| x.iter().filter_map(|y| y.as_ref().map(|&x| x)))
                    .zip(std::iter::repeat(x.0))
            })
            .map(|(x, y)| {
                self.interaction_e
                    [20 * self.mon[x].m_type() as usize + self.mon[y].m_type() as usize]
            })
            .fold(0.0, |acc, x| acc + x)
            * 0.5
    }
    fn eat(mut self, sweeps: usize, temp: Box<dyn Fn(f64) -> f64>, draw: bool, log: bool) -> Result<Vec<[f64; 4]>, Box<dyn Error>> {
        for _ in 0..self.mon.len().pow(2) {
            self.sweep()
        }
        for i in 0..sweeps {
            self.temperature = temp(i as f64 / sweeps as f64);
            self.sweep();
            self.save("data/outputdown", draw, log)?;
            // if i == 100 {
            //     panic!();
            // }
        }
        Ok(self.logvec)
    }
    fn save(&mut self, path: &str, draw: bool, log: bool) -> Result<(), Box<dyn Error>> {

        self.logvec.push([self.temperature, self.last_energy, self.end_to_end(), self.rog()]);

        if !log {
            return Ok(())
        }

        if let Some(mut old_file) = self.logger.take() {
            if draw {
                old_file.write_fmt(format_args!("\n{}\n", self))?;
            }

            old_file.write_fmt(format_args!("START: {}\n", self.sweeps()))?;
            old_file.write_fmt(format_args!("energy: {}\n", self.last_energy))?;
            old_file.write_fmt(format_args!("end_to_end: {}\n", self.last_distance))?;
            old_file.write_fmt(format_args!("temperature: {}\n", self.temperature))?;
            old_file.write_fmt(format_args!("rog: {}\n", self.rog()))?;
            old_file.write_fmt(format_args!("END: {}\n", self.sweeps()))?;


            self.logger = Some(old_file);

            Ok(())
        } else {
            std::fs::rename(format!("{}.txt", path), format!("{}_old.txt", path)).ok();
            let mut new_file = File::create_new(format!("{}.txt", path))?;

            for i in 0..20 {
                for j in 0..20 {
                    new_file.write_fmt(format_args!("{} ", self.interaction_e[i * 20 + j]))?;
                }
                new_file.write_fmt(format_args!("\n"))?;
            }

            self.logger = Some(new_file);
            self.save(path, draw, log)
        }
    }
    /// if who is None, updates everything.
    /// if Some(a), updates the neighbors of a, then a, then a's new neighbors.
    fn update_neighbors(&mut self, who: Option<usize>) {
        if let Some(a) = who {
            let indices = *self.mon[a].neighbors();
            for &chain_index in indices.iter().flat_map(|x| x.iter()).filter_map(|x| x.as_ref()) {
                self.update(chain_index)
            }
            self.update(a);
            let indices = *self.mon[a].neighbors();
            for &chain_index in indices.iter().flat_map(|x| x.iter()).filter_map(|x| x.as_ref()) {
                self.update(chain_index)
            }
        } else {
            for chain_index in 0..self.mon.len() {
                self.update(chain_index)
            }
        }
    }
    /// Updates the neighbors of a single monomer
    fn update(&mut self, chain_index: usize) {
        let mut new_neighbors = [[None; 2]; N];
        let pos = self.mon[chain_index].position();
        for i in 0..N {
            for j in 0..2 {
                new_neighbors[i][j] = self.mon
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != chain_index)
                .position(|(_, x)| {
                    x.position().iter().zip(pos.iter()).enumerate().all(|(w, x)| x.0 == x.1 || (w == i) && (*x.0 == *x.1 + 2 * j as i64 - 1))
                });
            }
        }
        self.mon[chain_index].neighbors = new_neighbors;
    }
    /// returns a vector over valid positions. Probably not ideal to collect so many vectors, but it works.
    /// the heavy use of closures and iterative programming here is to minimize unnecessary overhead. I apologize if it's a bit messy.
    fn valid_positions(&self, chain_index: usize) -> Vec<[i64; N]> {
        // returns an array over a given point's neighborss
        let gen = |x: [i64; N]| -> [[[i64; N]; 2]; N] {
            let mut ret = [[x; 2]; N];
            for i in 0..N {
                ret[i][0][i] -= 1;
                ret[i][1][i] += 1;
            }
            ret
        };
        // returns a vector over all empty neighbors of the input index if it is Some.
        let is_unavailable = |pos: Option<usize>| -> Option<Vec<[i64; N]>> {
            pos.map(|i| self.mon.get(i)).flatten().map(|x| {
                gen(x.position())
                    .into_iter()
                    .flat_map(|x| x.into_iter())
                    .filter(|&y| !self.mon.iter().any(|x| x.position() == y))
                    .collect()
            })
        };
        match (
            is_unavailable(chain_index.checked_sub(1)),
            is_unavailable(chain_index.checked_add(1)),
        ) {
            // if monomer not an edge, only returns positions both adjacent monomers neighbor.
            (Some(a), Some(b)) => a.into_iter().filter(|x| b.contains(x)).collect(),
            // if monomer is an edge, returns all available neighbors of the edge's neighbor
            (Some(a), None) | (None, Some(a)) => a,
            // monomer of length = 1. Not covered.
            _ => unreachable!(),
        }
    }
    fn new_linear(range_or_default: Option<usize>, n: usize, temp: f64) -> Self {
        let mut ret = Self::default();
        ret.mon = (0..n)
            .map(|i| {
                Monomer::new(
                    range_or_default.map_or(0, |x| random::<usize>() % x),
                    {
                        let mut ret = [0; N];
                        ret[0] = i as i64;
                        ret
                    },
                )
            })
            .collect();
        let mut rest = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                rest.push((((i * 5 + j) as f64) / 57.0) - 4.0)
            }
        }
        for i in 1..20 {
            for j in 0..i {
                rest[20 * i + j] = rest[20 * j + i]
            }
        }
        ret.temperature = temp;
        ret.interaction_e = rest;
        ret.update_neighbors(None);
        ret.last_energy = ret.energy();
        ret.sweep_size = n;
        ret
    }
}

impl Chain2d<2> {
    // fn _new_structure(m_type: usize, temp: f64, sweep: usize) -> Self {
    //     let mut ret = Self::default();
    //     let mut rest = Vec::new();
    //     for i in 0..20 {
    //         for j in 0..20 {
    //             rest.push((((i * 5 + j) as f64) / 57.0) - 4.0)
    //         }
    //     }
    //     for i in 1..20 {
    //         for j in 0..i {
    //             rest[20 * i + j] = rest[20 * j + i]
    //         }
    //     }
    //     ret.mon = vec![
    //         Monomer2d::new(m_type, [0, 0]),
    //         Monomer2d::new(m_type, [1, 0]),
    //         Monomer2d::new(m_type, [2, 0]),
    //         Monomer2d::new(m_type, [2, 1]),
    //         Monomer2d::new(m_type, [2, 2]),
    //         Monomer2d::new(m_type, [3, 2]),
    //         Monomer2d::new(m_type, [3, 1]),
    //         Monomer2d::new(m_type, [3, 0]),
    //         Monomer2d::new(m_type, [3, -1]),
    //         Monomer2d::new(m_type, [2, -1]),
    //         Monomer2d::new(m_type, [2, -2]),
    //         Monomer2d::new(m_type, [3, -2]),
    //         Monomer2d::new(m_type, [3, -3]),
    //         Monomer2d::new(m_type, [3, -4]),
    //         Monomer2d::new(m_type, [3, -5]),
    //         // Mynymer::new(m_type, [0, 0]),
    //         // Mynymer::new(m_type, [1, 0]),
    //         // Mynymer::new(m_type, [2, 0]),
    //         // Mynymer::new(m_type, [3, 0]),
    //         // Mynymer::new(m_type, [4, 0]),
    //         // Mynymer::new(m_type, [5, 0]),
    //         // Mynymer::new(m_type, [6, 0]),
    //         // Mynymer::new(m_type, [6, 1]),
    //         // Mynymer::new(m_type, [5, 1]),
    //         // Mynymer::new(m_type, [5, 2]),
    //         // Mynymer::new(m_type, [5, 3]),
    //         // Mynymer::new(m_type, [5, 4]),
    //         // Mynymer::new(m_type, [5, 5]),
    //         // Mynymer::new(m_type, [5, 6]),
    //         // Mynymer::new(m_type, [5, 7]),
    //         // Mynymer::new(m_type, [0, 0]),
    //         // Mynymer::new(m_type, [1, 0]),
    //         // Mynymer::new(m_type, [2, 0]),
    //         // Mynymer::new(m_type, [3, 0]),
    //         // Mynymer::new(m_type, [4, 0]),
    //         // Mynymer::new(m_type, [5, 0]),
    //         // Mynymer::new(m_type, [6, 0]),
    //         // Mynymer::new(m_type, [6, 1]),
    //         // Mynymer::new(m_type, [5, 1]),
    //         // Mynymer::new(m_type, [4, 1]),
    //         // Mynymer::new(m_type, [3, 1]),
    //         // Mynymer::new(m_type, [2, 1]),
    //         // Mynymer::new(m_type, [2, 2]),
    //         // Mynymer::new(m_type, [2, 3]),
    //         // Mynymer::new(m_type, [2, 4]),
    //     ];
    //     ret.interaction_e = rest;
    //     ret.update_neighbors(None);
    //     ret.last_energy = ret.energy();
    //     ret.temperature = temp;
    //     ret.sweep_size = sweep;
    //     ret
    // }
}