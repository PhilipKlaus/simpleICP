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
}

