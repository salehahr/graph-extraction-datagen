import os
import unittest
import cv2

from tools.PolyGraph import PolyGraph
from im2graph import (
    extract_nodes_and_edges,
    helper_polyfit,
    helper_structural_graph,
    polyfit_training,
    generate_graph
)


base_path = os.path.join(os.getcwd(), 'edges-0000_00400')


class TestExtractEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skeleton_fp = os.path.join(base_path, 'skeleton.png')
        cls.simple_skeleton_fp = os.path.join(base_path, 'simple-skeleton.png')
        cls.landmarks_fp = os.path.join(base_path, 'landmarks.png')
        cls.graph_fp = os.path.join(base_path, 'graph.json')

        cls.img_skel = cv2.imread(cls.skeleton_fp, cv2.IMREAD_GRAYSCALE)
        cls.img_simple_skel = cv2.imread(cls.simple_skeleton_fp, cv2.IMREAD_GRAYSCALE)
        cls.img_lm = cv2.cvtColor(cv2.imread(cls.landmarks_fp, cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        cls.graph = PolyGraph.load(cls.graph_fp)

        cls.num_edges = 18

    def test_extract_edges(self):
        nodes, edges, _ = extract_nodes_and_edges(self.img_skel)

        helper_pf_edges, _ = helper_polyfit(nodes, edges)
        helper_sg_edges, _ = helper_structural_graph(nodes, edges)

        polyfit_params = polyfit_training(helper_pf_edges)
        graph = generate_graph(helper_sg_edges, nodes, polyfit_params,
                               False, '')

        edges_extracted = edges['path']
        edges_in_graph = graph.edges

        self.assertEqual(len(edges_extracted), len(edges_in_graph))
