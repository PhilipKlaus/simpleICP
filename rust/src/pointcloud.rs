use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use linfa_linalg::eigh::{EighInto, EigSort};
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2, s};
use ndarray_stats::CorrelationExt;

use crate::nearest_neighbor::{knn_search, NNRes, NormalRes};

pub struct CloudToCloudDist {
    pub nn: Vec<Vec<NNRes>>,
    pub dist: Array1<f64>,
}

#[derive(Default)]
pub struct PointCloud {
    points: Array2<f64>,
    planarity: Array1<f64>,
    normals: Array2<f64>,
    selection: Option<Box<PointCloud>>,
    selected_idx: Vec<usize>,
}

//###############################
//# 'Static' PointCloud methods #
//###############################
impl PointCloud {
    pub fn new(points: Vec<f64>) -> PointCloud {
        let point_amount = points.len() / 3;
        PointCloud {
            points: Array::from_shape_vec((point_amount, 3), points)
                .expect("Could not create ndarray from points"),
            normals: Array::from_elem((point_amount, 3), f64::NAN),
            planarity: Array::from_elem(point_amount, f64::NAN),
            selection: None,
            selected_idx: (0..point_amount).collect(),
        }
    }

    pub fn select_from_cloud(cloud: &PointCloud, idx: &Vec<usize>) -> PointCloud {
        let new_point_amount = idx.len() / 3;
        PointCloud {
            points: cloud.points.select(Axis(0), idx),
            normals: cloud.normals.select(Axis(0), idx),
            planarity: cloud.planarity.select(Axis(0), idx),
            selection: None,
            selected_idx: (0..new_point_amount).collect(),
        }
    }

    pub fn read_from_xyz(path: &str) -> PointCloud {
        let file = File::open(path).expect("Could not read pointcloud from file");
        let reader = BufReader::new(file);
        let point_data = reader
            .lines()
            .into_iter()
            .map(|l| {
                let line = l.expect("Could not read line");
                let xyz: Vec<f64> = line.split_whitespace().map(|part| {
                    let coord: f64 = part.parse().expect("Unable to parse coordinate");
                    coord
                }).collect();
                xyz
            })
            .flatten()
            .collect();
        PointCloud::new(point_data)
    }

    pub fn write_to_file(cloud: &PointCloud, name: &str) {
        let file = File::create(name).expect("Could not open file");
        let mut writer = BufWriter::new(file);
        for pt in cloud.points().outer_iter() {
            write!(writer, "{} ", pt[[0]]).expect("Unable to write to file");
            write!(writer, "{} ", pt[[1]]).expect("Unable to write to file");
            write!(writer, "{}\n", pt[[2]]).expect("Unable to write to file");
        }
    }

    pub fn cloud_to_cloud_distance(pc1: &PointCloud, pc2: &PointCloud) -> CloudToCloudDist {
        let nn_res = knn_search(pc2, pc1, 1);
        let dists: Vec<f64> = pc1.points()
            .outer_iter()
            .zip(pc1.normals().outer_iter())
            .zip(nn_res.iter())
            .map(|((p1, n1), nn)| {
                let x1 = p1[[0]];
                let y1 = p1[[1]];
                let z1 = p1[[2]];

                let x2 = pc2.points()[[nn[0].idx, 0]];
                let y2 = pc2.points()[[nn[0].idx, 1]];
                let z2 = pc2.points()[[nn[0].idx, 2]];

                let nx1 = n1[[0]];
                let ny1 = n1[[1]];
                let nz1 = n1[[2]];
                (x2 - x1) * nx1 + (y2 - y1) * ny1 + (z2 - z1) * nz1
            }).collect();
        CloudToCloudDist {
            nn: nn_res,
            dist: Array1::from_vec(dists),
        }
    }
}

//###############################
//# 'Getters' for PointCloud    #
//###############################
impl PointCloud {
    pub fn point_amount(&self) -> usize {
        self.points.shape()[0]
    }

    pub fn points(&self) -> ArrayView<'_, f64, Ix2> {
        self.points.view()
    }

    pub fn planarity(&self) -> ArrayView<'_, f64, Ix1> {
        self.planarity.view()
    }

    pub fn normals(&self) -> ArrayView<'_, f64, Ix2> {
        self.normals.view()
    }

    pub fn selection(&self) -> &PointCloud {
        match self.selection {
            Some(ref x) => x,
            None => self
        }
    }

    pub fn selection_idx(&self) -> &Vec<usize> {
        &self.selected_idx
    }
}

//###############################
//#     PointCloud methods      #
//###############################

impl PointCloud {
    pub fn select_in_range(&mut self, cloud: &mut PointCloud, max_range: f64) {
        let now = Instant::now();
        let query = self.selection();

        // Get nearest neighbours
        let nn = knn_search(cloud, query, 1);

        let mut nn_iter = nn.iter();
        self.selected_idx.retain(|_| {
            let nn_item = nn_iter.next().unwrap();
            nn_item[0].distance <= max_range
        });

        self.selection = Option::from(Box::new(PointCloud::select_from_cloud(self, &self.selected_idx)));
        println!("select_in_range took: {}", now.elapsed().as_millis());
    }

    pub fn select_n_pts(&mut self, n: usize) {
        let now = Instant::now();
        // Todo: Build in check for "n"
        if n < self.selected_idx.len() {
            let dist_idx = Array::<f64, _>::linspace(0., (self.selected_idx.len() - 1) as f64, n);
            let mut dist_idx_iter = dist_idx.iter();
            let mut actual_idx = dist_idx_iter.next().unwrap().floor() as usize;

            let all_idx: Vec<usize> = (0..self.selected_idx.len()).collect();
            let mut all_idx_iter = all_idx.iter();

            self.selected_idx.retain(|_| {
                let next_idx = all_idx_iter.next().unwrap();
                if *next_idx == actual_idx {
                    match dist_idx_iter.next() {
                        None => {
                            actual_idx = usize::MAX;
                        }
                        Some(new_idx) => {
                            actual_idx = new_idx.floor() as usize;
                        }
                    }
                    true
                } else { false }
            });

            self.selection = Option::from(Box::new(PointCloud::select_from_cloud(self, &self.selected_idx)));
        }
        println!("select_n_pts took: {}", now.elapsed().as_millis());
    }

    pub fn estimate_normals(&mut self, neighbors: usize) {
        let now = Instant::now();
        let query_points = self.selection();

        let nn = knn_search(self, query_points, neighbors);

        self.normals = Array::from_elem((self.point_amount(), 3), f64::NAN);

        for (i, idx) in self.selected_idx.iter().enumerate() {
            let mut x_nn: Array<f64, Ix2> = Array::zeros((neighbors, 3));

            for j in 0..neighbors {
                x_nn[[j, 0]] = self.points[[nn[i][j].idx, 0]];
                x_nn[[j, 1]] = self.points[[nn[i][j].idx, 1]];
                x_nn[[j, 2]] = self.points[[nn[i][j].idx, 2]];
            }

            let normal = Self::normal_from_neighbors(&mut x_nn);
            self.normals[[*idx, 0]] = normal.eigenvector[[0]];
            self.normals[[*idx, 1]] = normal.eigenvector[[1]];
            self.normals[[*idx, 2]] = normal.eigenvector[[2]];
            self.planarity[[*idx]] = normal.planarity;
            // println!("nx1: {} | ny1: {} | nz1: {}", self.normals[[*idx, 0]], self.normals[[*idx, 1]], self.normals[[*idx, 2]]);

        }

        // If selection exists: partially update it
        if let Some(sel) = &mut self.selection {
            sel.normals = self.normals.select(Axis(0), &self.selected_idx);
            sel.planarity = self.planarity.select(Axis(0), &self.selected_idx);
        }
        println!("estimate_normals took: {}", now.elapsed().as_millis());
    }

    fn normal_from_neighbors(neighbors: &mut Array<f64, Ix2>) -> NormalRes {
        let covariance = neighbors.t().cov(1.).unwrap();
        let eig_res = covariance.eigh_into().unwrap();
        let mut eig_sort = eig_res.sort_eig_desc();
        let eig_vals = eig_sort.0.view();

        NormalRes {
            eigenvector: eig_sort.1.slice_mut(s![.., 2]).to_owned(),
            planarity: (eig_vals[1] - eig_vals[2]) / eig_vals[0],
        }
    }
}


#[cfg(test)]
mod point_cloud_test {
    use linfa_linalg::norm::Norm;
    use ndarray::{array, Array, Ix2};

    use crate::pointcloud::{NormalRes, PointCloud};

    fn get_points() -> Array<f64, Ix2> {
        array![
            [1., 1., 1.],
            [1., 2., 1.],
            [1., 3., 1.],
            [2., 1., 2.],
            [2., 2., 2.],
            [2., 3., 2.],
            [3., 1., 3.],
            [3., 2., 3.],
            [3., 3., 3.],
        ]
    }

    #[test]
    fn normals_from_neighbors() {
        let mut points = get_points();
        let normal = PointCloud::normal_from_neighbors(&mut points);
        let expected = array![1., 0., -1.];
        let norm = expected.norm_l2();

        let normalized = expected / norm;

        let delta = 0.01;
        assert_float_absolute_eq!(normal.planarity, 0.5, delta);
        assert_float_absolute_eq!(normalized[[0]], 0.7, delta);
        assert_float_absolute_eq!(normalized[[1]], 0.0, delta);
        assert_float_absolute_eq!(normalized[[2]], -0.7, delta);
    }
}
