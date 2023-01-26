use std::process::id;
use linfa_linalg::cholesky::{SolveC, SolveCInplace};

use linfa_linalg::svd::SVD;
use ndarray::{Array1, Array2, Zip};

use crate::pointcloud::PointCloud;

pub fn estimate_rigid_body_transformation(pc1: &PointCloud, pc2: &PointCloud) {
    let mut m_a: Array2<f64> = Array2::default((pc1.point_amount(), 6));
    let mut v_l: Array1<f64> = Array1::default((pc1.point_amount()));

    let all_idx: Vec<usize> = (0..pc1.point_amount()).collect();
    let mut idx = all_idx.into_iter();

    Zip::from(pc1.points().outer_iter())
        .and(pc1.normals().outer_iter())
        .and(pc2.points().outer_iter())
        .for_each(|p1, n1, p2| {
            let x_pc1 = p1[[0]];
            let y_pc1 = p1[[1]];
            let z_pc1 = p1[[2]];

            let nx_pc1 = n1[[0]];
            let ny_pc1 = n1[[1]];
            let nz_pc1 = n1[[2]];

            let x_pc2 = p2[[0]];
            let y_pc2 = p2[[1]];
            let z_pc2 = p2[[2]];

            let i = idx.next().unwrap();

            m_a[[i, 0]] = -z_pc2 * ny_pc1 + y_pc2 * nz_pc1;
            m_a[[i, 1]] = z_pc2 * nx_pc1 - x_pc2 * nz_pc1;
            m_a[[i, 2]] = -y_pc2 * nx_pc1 + x_pc2 * ny_pc1;
            m_a[[i, 3]] = nx_pc1;
            m_a[[i, 4]] = ny_pc1;
            m_a[[i, 5]] = nz_pc1;

            v_l[[i]] = nx_pc1 * (x_pc1 - x_pc2) + ny_pc1 * (y_pc1 - y_pc2) + nz_pc1 * (z_pc1 - z_pc2);
        });

    let svd = m_a.svd(true, true).expect("Could not calculate SVD");
    print!("Size of u: {:?}", svd.0.unwrap().shape());
    print!("Size of sigma: {:?}", svd.1.shape());
    print!("Size of uvt: {:?}", svd.2.unwrap().shape());

    //let res = svd.2.unwrap().solvec(v_l);
    /*let u = x_svd.0.unwrap();

    let alpha1 = u[[0]];
    let alpha2 = u[[1]];
    let alpha3 = u[[2]];
    let tx = u[[3]];
    let ty = u[[4]];
    let tz = u[[5]];*/
}