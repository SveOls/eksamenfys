#![feature(file_create_new)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
mod plotting;

use std::{error::Error, fmt::Display, fs::File, io::Write, sync::Arc, sync::Mutex, hash::{Hash, Hasher}};

use rand::random;

// dimensions of the lattice
const DIM: usize = 3;
// length of the chain (for new_linear).
const LEN: usize = 50;

fn main() -> Result<(), Box<dyn Error>> {
    tester();

    // variables all stored in one place for simplicity.
    let range = 20;
    let temperature = 20.0;
    let mintemp = 0.1;
    // margin adjusts the temperature functions. for sigmoid, it
    // increases the slope. for exponential, it makes the sweep cover a larger area,
    // clamped to the temperature range.
    let margin = 1.5f64;
    let sweeps = 10000;
    let sample_per_thread = 10;
    let threads = 10;
    let seed = 55;
    let cutoff: Option<f64> = Some(10.0);
    // let cutoff: Option<f64> = None;
    let chain: Chain<DIM, LEN> = Chain::new_linear(Some(range), temperature, seed);

    run2(temperature, 100, chain.clone())?;
    run(
        [mintemp, temperature],
        margin,
        sweeps,
        threads,
        sample_per_thread,
        cutoff,
        chain.clone(),
    )?;
    run3(
        [mintemp, temperature],
        margin,
        sweeps,
        cutoff,
        chain,
    )?;
    Ok(())
}

/// turns f64 and f32 into bits, increments or decrements the significand,
/// and turns them back into f64 and f32, adjusts exponent by subtracting 1.0, and
/// prints in scienfific notation.
fn tester() {
    println!("{:+e}", f64::from_bits(1.0f64.to_bits() + 1) - 1.0);
    println!("{:+e}", f64::from_bits(1.0f64.to_bits() - 1) - 1.0);
    println!("{:+e}", f32::from_bits(1.0f32.to_bits() + 1) - 1.0);
    println!("{:+e}", f32::from_bits(1.0f32.to_bits() - 1) - 1.0);
}

fn run<const N: usize, const L: usize>(
    temps: [f64; 2],
    margin: f64,
    sweeps: usize,
    threads: usize,
    sample_per_thread: usize,
    ljp: Option<f64>,
    mut chain: Chain<N, L>,
) -> Result<(), Box<dyn Error>>
where
    [(); 2 * N]:,
{
    chain.cutoff = ljp;
    // defines a function that will be given to each thread. Input is progress [0, 1],
    // and output is temperature
    let temp = move |inp: f64| -> f64 {
        (temps[1] - temps[0]) / ( 1.0 + (-10.0*margin * (0.5 - inp)).exp() ) + temps[0]
    };
    // This is an alternative to the function above. Where the one above is a sigmoid,
    // this is exponential - ie, it will slow down as it approaches zero.
    // let temp = move |inp: f64| -> f64 {
    //     (margin * temps[1] * (((temps[0] / margin.powi(2)) / temps[1]).ln() * inp).exp())
    //         .clamp(temps[0], temps[1])
    // };
    // i decided to use a sigmoid as the two areas that need the most sweeps are the start and the end,
    // with the center being fairly straight forward as long as equilibrium is reached before the sigmoid starts
    // to turn down.

    // data is wrapped in Arc<Mutex<_>>. This means it is single-access shared
    // data between all threads.
    let mut data = Arc::new(Mutex::new(vec![[0.0; 4]; sweeps]));
    // "counter" and "now" are for performance measurement purposes.
    let mut counter = 0;
    let now = std::time::Instant::now();
    // just a normal loop with no exit conditions, bar any errors.
    // p is used to appropriately merge data.
    for p in 0.. {
        // performance measurement prints
        println!("{}", counter);
        println!(
            "{}",
            counter as f64 * 1000.0 / now.elapsed().as_millis() as f64
        );
        // a vector over references to each thread. Will allow us to
        // wait for each thread to finish before using the data.
        let mut handles = Vec::new();
        for _ in 0..threads {
            let data = Arc::clone(&data);
            let chain = chain.clone();
            // spawns one thread for each run of the for loop
            let handle = std::thread::spawn(move || {
                // temporary data storage
                let mut ret = vec![[0.0; 4]; sweeps];
                for w in 0..sample_per_thread {
                    //
                    // all threads print their progress in the same place, but they're
                    // about as fast, so it works out.
                    print!("\r{:>6.1}%\r", 100.0 * w as f64 / sample_per_thread as f64);
                    std::io::stdout().flush().unwrap();
                    //
                    // clones the chain and consumes it, annealing it to T=mintemp.
                    let new = chain.clone()
                        .eat(sweeps, Box::new(temp), false, false, true)
                        .unwrap();
                    //
                    // two closures to avoid boilerplate.
                    // tldr: every data point is smoothed to 40% self,
                    // 20% each neighbor, 10% each second neighbor.
                    // If there are no neighbors there, it uses self.
                    let checked = |kl: Option<usize>, indexe: usize| -> Option<f64> {
                        kl.and_then(|t| new.get(t).map(|l| l[indexe]))
                    };
                    let tempe = |fe: usize, indexe: usize, y: f64| -> f64 {
                        (w as f64 * y + {
                            0.4 * new[fe][indexe]
                                + 0.20 * checked(fe.checked_sub(1), indexe).unwrap_or(new[fe][indexe])
                                + 0.1 * checked(fe.checked_sub(2), indexe).unwrap_or(new[fe][indexe])
                                + 0.20 * checked(fe.checked_add(1), indexe).unwrap_or(new[fe][indexe])
                                + 0.1 * checked(fe.checked_add(2), indexe).unwrap_or(new[fe][indexe])
                        }) / (w as f64 + 1.0)
                    };
                    //
                    // "ret" is temporary data storage.
                    ret.iter_mut().enumerate().for_each(|(i, x)| {
                        x[0] = new[i][0];
                        x[1] = tempe(i, 1, x[1]);
                        x[2] = tempe(i, 2, x[2]);
                        x[3] = tempe(i, 3, x[3]);
                    });
                    //
                }
                //
                // locks the main dataset - meaning this thread has sole access - and
                // updates it based on local data in the thread.
                let mut data = data.lock().unwrap();
                data.iter_mut().zip(ret.into_iter()).for_each(|(x, y)| {
                    x[0] = y[0];
                    x[1] = (p as f64 * x[1] + y[1]) / ((p + 1) as f64);
                    x[2] = (p as f64 * x[2] + y[2]) / ((p + 1) as f64);
                    x[3] = (p as f64 * x[3] + y[3]) / ((p + 1) as f64);
                });
                //
            });
            handles.push(handle);
        }
        //
        // halts the main thread until all threads are complete
        for handle in handles {
            handle.join().unwrap();
        }
        //
        // unwraps the arc<mutex<_>>. As we already waited until all threads
        // were finished, this won't crash.
        let data2 = Arc::<Mutex<Vec<[f64; 4]>>>::try_unwrap(data)
            .unwrap()
            .into_inner()?;
        //
        // plots diagrams for each of the three variables,
        // as a phase diagram and versus sweep.
        plotting::doubledraw(&data2, 1, "1energy_ljp.png", None)?;
        plotting::phasedraw(&data2, 1,  "1energyphase_ljp.png", None)?;
        plotting::doubledraw(&data2, 2, "1etoe_ljp.png", None)?;
        plotting::phasedraw(&data2, 2,  "1etoephase_ljp.png", None)?;
        plotting::doubledraw(&data2, 3, "1rog_ljp.png", None)?;
        plotting::phasedraw(&data2, 3,  "1rogphase_ljp.png", None)?;
        //
        // rewraps the data into arc<mutex<_>> for the next pass.
        data = Arc::new(Mutex::new(
            data2
        ));
        //
        // performance logging
        counter += L * sample_per_thread * threads * (sweeps + 5 * L.pow(2));
        //
    }
    Ok(())
}

/// runs a single polymer (sweep) times at temperature (temp).
fn run2<const N: usize, const L: usize>(
    temp: f64,
    sweeps: usize,
    chain: Chain<N, L>,
) -> Result<(), Box<dyn Error>>
where
    [(); 2 * N]:,
{
    // always returns temperature. Allows us to reuse the same function
    // as the other function.
    let temp = move |_: f64| -> f64 { temp };
    let data = chain.eat(sweeps, Box::new(temp), true, true, false)?;
    //
    // plotting
    plotting::singledraw(&data, 1, "out1b.png", None)?;
    plotting::singledraw(&data, 2, "out2b.png", None)?;
    plotting::singledraw(&data, 3, "out3b.png", None)?;
    //
    Ok(())
}

/// runs a single polymer (sweep) times at temperature (temp).
fn run3<const N: usize, const L: usize>(
    temps: [f64; 2],
    margin: f64,
    sweeps: usize,
    ljp: Option<f64>,
    mut chain: Chain<N, L>,
) -> Result<(), Box<dyn Error>>
where
    [(); 2 * N]:,
{
    chain.cutoff = ljp;
    // always returns temperature. Allows us to reuse the same function
    // as the other function.
    let temp = move |inp: f64| -> f64 {
        (temps[1] - temps[0]) / ( 1.0 + (-10.0*margin * (0.5 - inp)).exp() ) + temps[0]
    };
    let data = chain.eat(sweeps, Box::new(temp), true, true, false)?;
    //
    // plotting
    plotting::singledraw(&data, 1, "annealenergy.png", None)?;
    plotting::singledraw(&data, 2, "annealendtoend.png", None)?;
    plotting::singledraw(&data, 3, "annealrog.png", None)?;
    //
    Ok(())
}

/// A monomer. has a m_type [0..20], position, and
/// an array over option(neighbors). It uses const generic to
/// allow for an arbitrary number of dimensions while remaining
/// on the stack (for performance reasons).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Monomer<const N: usize>
where
    [(); 2 * N]:,
{
    m_type: usize,
    position: [i64; N],
    neighbors: [Option<usize>; 2 * N],
}
impl<const N: usize> Monomer<N>
where
    [(); 2 * N]:,
{
    /// generates a new monomer from input
    pub fn new(m_type: usize, position: [i64; N]) -> Self {
        Self {
            m_type,
            position,
            neighbors: [None; 2 * N],
        }
    }
}

/// the core of the implementation. Chain has two const variables, one to
/// show dimensions (N) and one to show polymer length (L).
#[derive(Debug)]
struct Chain<const N: usize, const L: usize>
where
    [(); 2 * N]:,
{
    monomers: [Monomer<N>; L],
    interaction_e: Vec<f64>,
    last_energy: f64,
    temperature: f64,
    logger: Option<File>,
    logvec: Vec<[f64; 4]>,
    itercount: usize,
    sweep_size: usize,
    sigma: f64,
    cutoff: Option<f64>,
}

// manual implementation of Clone, as File has no implentation.
// this means we avoid making the same chain multiple times, and can just
// make it once for cleaner code.
impl<const N: usize, const L: usize> Clone for Chain<N, L>
where
    [(); 2 * N]:,
{
    fn clone(&self) -> Self {
        Self {
            monomers: self.monomers.clone(),
            interaction_e: self.interaction_e.clone(),
            last_energy: self.last_energy,
            temperature: self.temperature,
            logger: None,
            logvec: self.logvec.clone(),
            itercount: self.itercount,
            sweep_size: self.sweep_size,
            sigma: self.sigma,
            cutoff: self.cutoff.clone(),
        }
    }
}

/// in rust, iterators are very useful. By treating chain as an iterator,
/// where each step forward equals one draw, we can use utility functions like
/// .nth(), we can use Chain as the object of a for loop, and more.
impl<const N: usize, const L: usize> Iterator for Chain<N, L>
where
    [(); 2 * N]:,
{
    // iterators return None when complete. If it returns Some(Item),
    // we know we can keep going. In this case, Item is None when we fail a
    // draw for whatever reason, and Some(id) when we move a monomer.
    // A successful draw is thus Some(Some(id)), a failed draw is
    // Some(None), and a finished iterator is None. This one has exit condition, so
    // it will keep going forever as long as it doesn't crash.
    type Item = Option<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        self.itercount += 1;

        //
        // picks a random monomer
        let test_index = random::<usize>() % L;
        //
        // finds a list of possible moves for the monomer
        let val = self.valid_positions(test_index);
        if val.is_empty() {
            return Some(None);
        }
        // if there are possible moves, pick one.
        let pos = val[random::<usize>() % val.len()];
        //
        // save old data in case the move fails
        let old_pos = self.monomers[test_index].position;
        //
        // moves monomer to new position
        let tempe = self.energy(test_index);
        self.monomers[test_index].position = pos;
        let tempe = tempe - self.energy(test_index);
        //
        // (jehova's witness activities)
        // updates the old and new neighbors
        if self.cutoff.is_none() {
            self.update_neighbors(Some(test_index));
        }
        //
        // saved to variable to avoid double calculations
        // let energy = self.full_energy();
        //
        // if min(exp(Delta(E) / T), 1) > random[0, 1]
        if ((tempe) / self.temperature)
            .exp()
            .min(1.0)
            > random::<f64>()
        {
            //
            // saving variables
            self.last_energy -= tempe;
            //
            // the last line in each code block has an implicit "return", so
            // this is one possible return value for the function
            Some(Some(test_index))
            //
        } else {
            //
            // if the move fails, restore old position
            self.monomers[test_index].position = old_pos;
            if self.cutoff.is_none() {
                self.update_neighbors(Some(test_index));
            }
            Some(None)
            //
        }
    }
}

/// extremely ugly implementation of the train Display, letting me
/// print the polymer. Only prints 2D.
/// i recommend pressing the "collapse" button.
impl<const N: usize, const L: usize> Display for Chain<N, L>
where
    [(); 2 * N]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if N != 2 {
            return Ok(());
        }
        let max = self.monomers.iter().fold([i64::MAX, i64::MIN], |acc, x| {
            [acc[0].min(x.position[0]), acc[1].max(x.position[0])]
        });
        let may = self.monomers.iter().fold([i64::MAX, i64::MIN], |acc, x| {
            [acc[0].min(x.position[1]), acc[1].max(x.position[1])]
        });
        let maymin = if may[0] < 1 { 1 } else { 0 };
        let maxmin = if max[0] < 1 { 1 } else { 0 };
        let ywidth = ((may[1].abs().max(1).ilog10())
            .max((may[0] - 1).abs().max(1).ilog10() + maymin)
            + 2) as usize;
        let xwidth = ((max[1].abs().max(1).ilog10())
            .max((max[0] - 1).abs().max(1).ilog10() + maxmin)
            + 2) as usize;
        for y in (may[0] - 1..=may[1] + 1).rev() {
            write!(f, "{:1$} |", y, ywidth)?;
            for x in max[0] - 1..=max[1] + 1 {
                if let Some(a) = self
                    .monomers
                    .iter()
                    .position(|z| z.position[0..2] == [x, y])
                {
                    if a == 0 {
                        if self.monomers[a + 1].position[0] == self.monomers[a].position[0] + 1
                        {
                            write!(f, "{:2$}{:━<3$}", "", "S", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        } else if self.monomers[a + 1].position[0]
                            == self.monomers[a].position[0] - 1
                        {
                            write!(f, "{:━<2$}{:3$}", "", "S", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        } else {
                            write!(f, "{:^1$}", "S", xwidth)?;
                        }
                    } else if a == self.monomers.len() - 1 {
                        if self.monomers[a - 1].position[0] == self.monomers[a].position[0] + 1
                        {
                            write!(f, "{:2$}{:━<3$}", "", "E", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        } else if self.monomers[a - 1].position[0]
                            == self.monomers[a].position[0] - 1
                        {
                            write!(f, "{:━<2$}{:3$}", "", "E", (xwidth - 1) / 2, xwidth / 2 + 1)?;
                        } else {
                            write!(f, "{:^1$}", "E", xwidth)?;
                        }
                    } else if self.monomers[a - 1].position[0]
                        == self.monomers[a + 1].position[0]
                    {
                        write!(f, "{:^1$}", "┃", xwidth)?;
                    } else if self.monomers[a - 1].position[1]
                        == self.monomers[a + 1].position[1]
                    {
                        write!(
                            f,
                            "{:━^2$}{:━<3$}",
                            "",
                            "━",
                            (xwidth - 1) / 2,
                            xwidth / 2 + 1
                        )?;
                    } else if self.monomers[a - 1].position[1]
                        == self.monomers[a].position[1] + 1
                    {
                        if self.monomers[a + 1].position[0] == self.monomers[a].position[0] + 1
                        {
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
                    } else if self.monomers[a + 1].position[1]
                        == self.monomers[a].position[1] + 1
                    {
                        if self.monomers[a - 1].position[0] == self.monomers[a].position[0] + 1
                        {
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
                    } else if self.monomers[a - 1].position[1]
                        == self.monomers[a].position[1] - 1
                    {
                        if self.monomers[a + 1].position[0] == self.monomers[a].position[0] - 1
                        {
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
                    } else if self.monomers[a + 1].position[1]
                        == self.monomers[a].position[1] - 1
                    {
                        if self.monomers[a - 1].position[0] == self.monomers[a].position[0] - 1
                        {
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

// member functions of Chain are defined here
impl<const N: usize, const L: usize> Chain<N, L>
where
    [(); 2 * N]:,
{
    ///
    /// calculates the number of completed sweeps based on
    /// internal data
    fn sweeps(&self) -> usize {
        self.itercount / self.sweep_size
    }
    ///
    /// as we have implemented Iterator for Chain, self.nth(i) lets us draw
    /// i + 1 times.
    fn sweep(&mut self) {
        self.nth(self.sweep_size - 1);
    }
    ///
    /// distance between ends of polymer
    fn end_to_end(&self) -> f64 {
        let fpos = self.monomers.first().map(|x| x.position).unwrap();
        let lpos = self.monomers.last().map(|x| x.position).unwrap();
        fpos.iter()
            .zip(lpos.iter())
            .fold(0.0, |acc, z| acc + (*z.0 - *z.1).pow(2) as f64)
            .sqrt()
    }
    ///
    /// first calculates the center, then returns the sum over
    /// mean distance squared over each axis for each monomer.
    /// it's not pretty code, but it was the source of at least
    /// half my bugs, so i'm afraid to make it easier on the eyes.
    fn rog(&self) -> f64 {
        //
        // calculates center position
        let center = self
            .monomers
            .iter()
            .map(|x| x.position)
            .fold([0.0f64; N], |acc, x| {
                let mut ret = [0.0; N];
                for i in 0..N {
                    ret[i] = acc[i] + (x[i] as f64 / L as f64)
                }
                ret
            });
        let mut ret = 0.0;
        //
        // for each monomer, find average of RoG for each axis.
        for i in self.monomers.iter().map(|x| x.position) {
            ret += center
                .iter()
                .zip(i.iter())
                .fold(0.0, |acc, x| acc + (x.0 - *x.1 as f64).powi(2))
                / N as f64
        }
        // implicit return keyword
        ret
    }
    ///
    /// iterates through every monomer, and turns them into an iterator over connections.
    /// as each connection appears twice, the final product is divided by 2.
    /// self.interaction_e is the 20x20 matrix of interaction energies.
    ///
    /// ljp energy
    fn full_energy(&self) -> f64 {
        if let Some(cutoff) = self.cutoff {
            self.monomers
                .iter()
                .enumerate()
                .flat_map(|(i, x)| {
                    self.monomers
                        .iter()
                        .enumerate()
                        .filter(move |(j, _)| j.abs_diff(i) > 1)
                        .zip(std::iter::repeat(x))
                })
                .map(|((_, x), y)| (x.m_type, y.m_type, y.position.iter().zip(x.position.iter()).fold(0.0f64, |acc, x| acc.hypot((x.1 - x.0) as f64))))
                .filter(|(_, _, z)| *z <= cutoff)
                .map(|(x, y, z)| {
                    self.interaction_e[20 * x + y] * 4.0 * ((self.sigma / z).powi(12) - (self.sigma / z).powi(6))
                })
                .fold(0.0, |acc, x| acc - x)
                * 0.5
        } else {
            self.monomers
                .iter()
                .enumerate()
                .flat_map(|x| {
                    x.1.neighbors
                        .iter()
                        .filter_map(|x| x.as_ref())
                        .zip(std::iter::repeat(x.0))
                })
                .map(|(&x, y)| {
                    self.interaction_e[20 * self.monomers[x].m_type + self.monomers[y].m_type]
                })
                .fold(0.0, |acc, x| acc + x)
                * 0.5
        }
    }
    ///
    /// iterates through every monomer, and turns them into an iterator over connections.
    /// as each connection appears twice, the final product is divided by 2.
    /// self.interaction_e is the 20x20 matrix of interaction energies.
    ///
    /// ljp energy
    fn energy(&self, inp: usize) -> f64 {
        if let Some(cutoff) = self.cutoff {
            self.monomers
                .iter()
                .enumerate()
                .filter(|(j, _)| j.abs_diff(inp) > 1)
                .map(|(_, x)| (x.m_type, self.monomers[inp].m_type, self.monomers[inp].position.iter().zip(x.position.iter()).fold(0.0f64, |acc, x| acc.hypot((x.1 - x.0) as f64))))
                .filter(|(_, _, z)| *z <= cutoff)
                .map(|(x, y, z)| {
                    self.interaction_e[20 * x + y] * 4.0 * ((self.sigma / z).powi(12) - (self.sigma / z).powi(6))
                })
                .fold(0.0, |acc, x| acc - x)
        } else {
            self.monomers[inp]
                .neighbors.iter().filter_map(|x| x.as_ref())

                .map(|&x| {
                    self.interaction_e[20 * self.monomers[x].m_type + self.monomers[inp].m_type]
                })
                .fold(0.0, |acc, x| acc + x)
        }
    }
    ///
    /// consumes self, sweeps "sweeps" times, adjusts temperature based on "temp",
    /// returns data of the run.
    fn eat(
        mut self,
        sweeps: usize,
        temp: Box<dyn Fn(f64) -> f64>,
        draw: bool,
        log: bool,
        prerun: bool,
    ) -> Result<Vec<[f64; 4]>, Box<dyn Error>> {
        self.last_energy = self.full_energy();
        //
        // if the caller wants a prerun, sweep 5n times.
        if prerun {
            for _ in 0..self.monomers.len().pow(2) {
                self.sweep()
            }
        self.itercount = 0;
        }
        //
        // resets datalog, initialized by NaN.
        self.logvec = vec![[f64::NAN; 4]; sweeps];
        //
        // sweeps (sweeps) times
        for i in 0..sweeps {
            //
            // updates temperature based on the input function,
            // sweeps, and saves. If log is true, write to file - otherwise,
            // just update internal datalog. If draw (and log) is true,
            // prints the chain in the text file alongside the data.
            self.temperature = temp(i as f64 / sweeps as f64);
            self.sweep();
            self.save("data/outputdown", draw, log)?;
        }
        //
        // returns the datalog, drops the rest of the chain
        Ok(self.logvec)
        //
    }
    ///
    /// Updates internal log.
    /// if log is true, write to text.
    /// if draw is also true, include a drawing of the polymer
    /// (only if N=2).
    fn save(&mut self, path: &str, draw: bool, log: bool) -> Result<(), Box<dyn Error>> {
        //
        // saves ind to external variable, because you can't
        // both change Self while also reading Self. Quirk of rust.
        let ind = self.sweeps() - 1;
        self.logvec[ind] = [
            self.temperature,
            self.last_energy,
            self.end_to_end(),
            self.rog(),
        ];

        //
        // internal logger has been updated.
        // if external log isn't to be updated,
        // return.
        if !log {
            return Ok(());
        }
        //
        // if this is NOT the first external log, enter this.
        if let Some(mut old_file) = self.logger.take() {
            if draw {
                old_file.write_fmt(format_args!("\n{}\n", self))?;
            }

            old_file.write_fmt(format_args!("START: {}\n", self.sweeps()))?;
            old_file.write_fmt(format_args!("energy: {}\n", self.last_energy))?;
            old_file.write_fmt(format_args!("end_to_end: {}\n", self.end_to_end()))?;
            old_file.write_fmt(format_args!("temperature: {}\n", self.temperature))?;
            old_file.write_fmt(format_args!("rog: {}\n", self.rog()))?;
            old_file.write_fmt(format_args!("END: {}\n", self.sweeps()))?;

            self.logger = Some(old_file);

            Ok(())
        } else {
            //
            // external log file hasn't been created, so:
            // first, rename any existing file to avoid accidental deleting of data.
            std::fs::rename(format!("{}.txt", path), format!("{}_old.txt", path)).ok();
            //
            // then, create a new file in the same place. Will halt / crash the program if
            // the file already exists - but we just moved it, so it won't.
            let mut new_file = File::create_new(format!("{}.txt", path))?;
            //
            // since this is the first time this external log is used, we start
            // by writing down the interaction matrix.
            for i in 0..20 {
                for j in 0..20 {
                    new_file.write_fmt(format_args!("{} ", self.interaction_e[i * 20 + j]))?;
                }
                new_file.write_fmt(format_args!("\n"))?;
            }
            //
            // saves the File object into Self.
            self.logger = Some(new_file);
            //
            // recursive save - however, it can't recurse more than once, as we just updated self.logger.
            // it will update the same index for the internal logger twice, but it's not a problem.
            // this time, the if let-statement will be true, so the other block is evaluated.
            self.save(path, draw, log)
        }
    }
    /// if who is None, updates everything - meant for initialization.
    /// if Some(a), updates the neighbors of a, then a, then a's new neighbors - meant for single-monomer moves.
    fn update_neighbors(&mut self, who: Option<usize>) {
        //
        // if we're just updating a single index:
        if let Some(updated_index) = who {
            //
            // updates all of updated_index's neighbors
            let indices = self.monomers[updated_index].neighbors;
            for &chain_index in indices.iter().filter_map(|x| x.as_ref()) {
                self.update(chain_index)
            }
            //
            // updates updated_index
            self.update(updated_index);
            //
            // then updates all of updated_index's new neighbors
            let indices = self.monomers[updated_index].neighbors;
            for &chain_index in indices.iter().filter_map(|x| x.as_ref()) {
                self.update(chain_index)
            }
        } else {
            //
            // updates ALL monomers.
            for chain_index in 0..self.monomers.len() {
                self.update(chain_index)
            }
        }
    }
    ///
    /// Updates the neighbors of a single monomer
    fn update(&mut self, chain_index: usize) {
        //
        // resets the neighbor list
        let mut new_neighbors = [None; 2 * N];
        //
        // saves the position of the neighbor
        let pos = self.monomers[chain_index].position;
        //
        // double loop; i refers to the axis, j to the direction.
        // loops exactly 2N times.
        for i in 0..N {
            for j in 0..2 {
                //
                // applies .position() to an enumerated list of all monomers. This means that
                // every monomer will be evaluated to some criteria, and the first criteria that
                // returns True, will return Some(index). If none fit the criteria, it returns None.
                //
                // the criteria is a logic block that must evaluate to True on all axes.
                // it boils down to this: the coordinates must be the exact same, except
                // for on the i axis (the outermost loop). on that position, the coordinate
                // must differ by 1 (in a negative direction for j==0, and positive for j==1).
                new_neighbors[2 * i + j] = self.monomers.iter().enumerate().position(|(l, x)| {
                    x.position
                        .iter()
                        .zip(pos.iter())
                        .enumerate()
                        .all(|(w, x)| {
                            x.0 == x.1 || (w == i) && (*x.0 == ((*x.1 + 2 * j as i64) - 1))
                        })
                        && l.abs_diff(chain_index) > 1
                });
                //
            }
        }
        self.monomers[chain_index].neighbors = new_neighbors;
    }
    ///
    /// returns a vector over valid positions. Probably not ideal to collect so many vectors, but it works.
    /// the heavy use of closures and iterative programming here is to minimize unnecessary overhead. I apologize if it's a bit messy.
    fn valid_positions(&self, chain_index: usize) -> Vec<[i64; N]> {
        //
        // a closure defined on its own for simplicity
        // returns an array over a given point's neighborss
        let gen = |x: [i64; N]| -> [[i64; N]; 2 * N] {
            let mut ret = [x; 2 * N];
            for i in 0..N {
                ret[2 * i][i] -= 1;
                ret[2 * i + 1][i] += 1;
            }
            ret
        };
        //
        // returns a vector over all empty neighbor slots of the input index if it is Some.
        let is_unavailable = |pos: Option<usize>| -> Option<Vec<[i64; N]>> {
            pos.and_then(|i| self.monomers.get(i)).map(|x| {
                gen(x.position)
                    .into_iter()
                    .filter(|&y| !self.monomers.iter().any(|x| x.position == y))
                    .collect()
            })
        };
        //
        // if both sides of the monomer exist, only return a list of options they have in common.
        // else, just return one of them.
        // if there are no neighbors (ie, chain length of 1) the program will crash.
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
    ///
    /// generates an unfolded, straight chain.
    fn new_linear(range_or_default: Option<usize>, temp: f64, seed: usize) -> Self {
        //
        // "range" refers to number of different monomers.
        // if input is None, assumes all monomers are identical.cc
        let range = range_or_default.unwrap_or(1);
        //
        // for every position in the chain, generate a number from position and seed.
        // it is a hash function, so if the seed is the same, it will return the exact
        // same chain.
        let monomers = core::array::from_fn(|inp: usize| -> Monomer<N> {
            Monomer::new({
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                (seed, inp).hash(&mut hash);
                hash.finish() as usize
            }%range, {
                let mut monomer_position = [0; N];
                monomer_position[0] = inp as i64;
                monomer_position
            })
        });
        let mut interaction_e = vec![f64::NAN; range.pow(2)];
        let valrange = [-4.0, -2.0];
        //
        // generates interaction energy matrix. appropriately symmetrical (M[m, n] = M[n, m])
        for i in 0..range {
            for j in 0..range {
                let val = (range + 1) as f64 / range as f64 * (((i+j)%range) as f64 / range as f64) * (valrange[0] - valrange[1]) + valrange[1];
                interaction_e[range * j.min(i) + j.max(i)] = val;
                interaction_e[range * j.max(i) + j.min(i)] = val;
            }
        }
        let mut linear_chain = Self {
            temperature: temp,
            interaction_e,
            monomers,
            last_energy: 0.0,
            logger: None,
            logvec: Vec::new(),
            itercount: 0,
            sweep_size: L,
            sigma: 2.0f64.powf(6.0f64.recip()).recip(),
            cutoff: None,
        };
        //
        // updates neighbors. does nothing, as it's linear, but it's
        // here for completeness.
        linear_chain.update_neighbors(None);
        linear_chain
    }
}

impl<const L: usize> Chain<2, L> {
    // fn _new_structure(m_type: usize, temp: f64, sweep: usize) -> Self {
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
    //         Monomer::new(m_type, [0, 0]),
    //         Monomer::new(m_type, [1, 0]),
    //         Monomer::new(m_type, [2, 0]),
    //         Monomer::new(m_type, [2, 1]),
    //         Monomer::new(m_type, [2, 2]),
    //         Monomer::new(m_type, [3, 2]),
    //         Monomer::new(m_type, [3, 1]),
    //         Monomer::new(m_type, [3, 0]),
    //         Monomer::new(m_type, [3, -1]),
    //         Monomer::new(m_type, [2, -1]),
    //         Monomer::new(m_type, [2, -2]),
    //         Monomer::new(m_type, [3, -2]),
    //         Monomer::new(m_type, [3, -3]),
    //         Monomer::new(m_type, [3, -4]),
    //         Monomer::new(m_type, [3, -5]),
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
