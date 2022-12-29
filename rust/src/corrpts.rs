use std::num::FpCategory::Nan;
use ndarray::{Array, Array1, Axis};
use crate::pointcloud::PointCloud;

// pc1 == fixed / pc2 == moved
pub fn match_point_clouds(pc1: &PointCloud, pc2: &PointCloud) {
    let pts_sel_pc1 = pc1.get_selected_points();
    let mut pts_sel_pc2 = pc2.get_selected_points();

    let idx_pc1 = pc1.get_idx_of_selected_points();
    let mut idx_pc2: Vec<usize> = vec![];

    let nn = PointCloud::knn_search(&mut pts_sel_pc2, &pts_sel_pc1, 1);

    for nn_res in nn.iter() {
        idx_pc2.push(nn_res[0].item.id);
    }

    let planarity = pc1.planarity.select(Axis(0), &idx_pc1);
    dist_between_clouds(pc1, pc2, &idx_pc1, &idx_pc2);
}

fn dist_between_clouds(pc1: &PointCloud, pc2: &PointCloud, idx1: &Vec<usize>, idx2: &Vec<usize>) -> Array1<f64> {
    let mut dists: Array1<f64> = Array::from_elem((idx1.len()), f64::NAN);
    for(idx, (i1, i2)) in idx1.iter().zip(idx2.iter()).enumerate() {
        let x1 = pc1.points[*i1].point[0];
        let y1 = pc1.points[*i1].point[1];
        let z1 = pc1.points[*i1].point[2];

        let x2 = pc2.points[*i2].point[0];
        let y2 = pc2.points[*i2].point[1];
        let z2 = pc2.points[*i2].point[2];

        let nx1 = pc1.normals[[*i1, 0]];
        let ny1 = pc1.normals[[*i1, 1]];
        let nz1 = pc1.normals[[*i1, 2]];

        dists[[idx]] = (x2 - x1) * nx1 + (y2 - y1) * ny1 + (z2 - z1) * nz1;
    }
    dists
}