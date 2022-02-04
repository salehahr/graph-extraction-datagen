import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tests.test_images import RandomImage
from tools.images import normalise
from tools.NodeContainer import NodeContainer
from tools.NodeExtractor import NodeExtractor
from tools.plots import node_types_image, plot_bgr_img
from tools.Point import distance, four_connectivity, positive_neighbours

data_path = os.path.join(os.getcwd(), "../data/test")


class TestNodeExtraction(RandomImage):
    @classmethod
    def setUpClass(cls) -> None:
        random = True

        if random:
            super(TestNodeExtraction, cls).setUpClass()
        if not random:
            base_path = os.path.join(data_path, "extract_nodes-GRK014-0002-08360")
            cls.img_skeletonised_fp = os.path.join(base_path, "skeleton.png")
            cls.title = os.path.relpath(cls.img_skeletonised_fp, start=data_path)

        cls.img_skel = cv2.imread(cls.img_skeletonised_fp, cv2.IMREAD_GRAYSCALE)
        # cls.plot_skeletonised()

        l_crossing_nodes, l_end_nodes, l_skel = cls.legacy_node_extraction()
        cls.legacy_nodes = NodeContainer(l_crossing_nodes, l_end_nodes, [])
        cls.legacy_skel = normalise(l_skel)

    @classmethod
    def plot_skeletonised(cls):
        plot_bgr_img(cls.img_skel, title=cls.title)
        plt.show()

    def test_node_extractor(self):
        ne = NodeExtractor(self.img_skel)
        self._compare(ne.img, ne.nodes)

    def _compare(self, skel: np.ndarray, nodes: NodeContainer):
        skel = normalise(skel)
        self._plot_comparison(skel, nodes)

        self.assertEqual(nodes.crossing_nodes_yx, self.legacy_nodes.crossing_nodes_yx)
        self.assertEqual(nodes.end_nodes_yx, self.legacy_nodes.end_nodes_yx)
        np.testing.assert_equal(skel, self.legacy_skel)

    def _plot_comparison(
        self,
        skel_new: np.ndarray,
        nodes_new: NodeContainer,
    ) -> None:
        plt.subplot(1, 3, 1)
        plot_bgr_img(self.legacy_skel)
        plt.title("Legacy")

        plt.subplot(1, 3, 2)
        plot_bgr_img(skel_new)
        plt.title("New")

        plt.subplot(1, 3, 3)
        plot_bgr_img(np.abs(self.legacy_skel - skel_new))
        plt.title("Difference")

        plt.suptitle(self.title)
        plt.show()

        node_img_legacy = node_types_image(self.legacy_nodes, skeleton=self.legacy_skel)
        node_img_new = node_types_image(nodes_new, skeleton=skel_new)
        plt.subplot(1, 2, 1)
        plot_bgr_img(node_img_legacy)
        plt.title("Legacy")

        plt.subplot(1, 2, 2)
        plot_bgr_img(node_img_new)
        plt.title("New")

        plt.suptitle(self.title)
        plt.show()

    @classmethod
    def legacy_node_extraction(
        cls,
    ) -> Tuple[List[List[int]], List[List[int]], np.ndarray]:
        cleaned_skeleton = cls.img_skel
        node_thick = 3

        skeleton = cleaned_skeleton.copy()
        skeleton_img = skeleton.copy()
        kernel = 3
        bc = []
        endpoints = []
        binary = skeleton_img.copy()
        binary[binary == 255] = 1
        result = skeleton_img.copy()
        n = int(np.floor(kernel / 2))
        for it_x in range(n, binary.shape[0] - n):
            for it_y in range(n, binary.shape[1] - n):
                neighbours_nodeall = []
                neighbours_nn = []
                bo = []
                cross = []
                aux = 0
                if binary[it_x, it_y] == 1:
                    aux += np.sum(
                        binary[it_x - n : it_x + n + 1, it_y - n : it_y + n + 1]
                    )  # Anzahl der Pixel mit 1 in der neighbourhood werden gezählt (inkl. Mittelpunkt)
                    if aux == 2:  # endpoint
                        endpoints.append([it_x, it_y])
                    if aux == 3:  # endpoint bei 90° Winkel
                        neighbours_nodeall = positive_neighbours(it_x, it_y, binary)
                        conn = four_connectivity(
                            neighbours_nodeall[0][0], neighbours_nodeall[0][1]
                        )
                        if neighbours_nodeall[1] in conn:
                            endpoints.append([it_x, it_y])
                    if (
                        aux == 4
                    ):  # Vergabelung = 4 Pixel mit 1 -> Punkt wird gelöscht, Koordinaten werden gespeichert
                        neighbours_nodeall = positive_neighbours(it_x, it_y, binary)
                        for q in range(0, len(neighbours_nodeall)):
                            neighbours_nn.append(
                                four_connectivity(
                                    neighbours_nodeall[q][0], neighbours_nodeall[q][1]
                                )
                            )
                        for p in range(0, len(neighbours_nodeall)):
                            for j in range(0, len(neighbours_nn)):
                                if neighbours_nodeall[p] in neighbours_nn[j]:
                                    bo.append(True)
                                else:
                                    bo.append(False)
                        if not any(bo):
                            result[it_x, it_y] = 0
                            bc.append([it_x, it_y])
                    elif aux >= 5:  # Vergabelung oder Kreuzung
                        neighbours_nodeall = positive_neighbours(it_x, it_y, binary)
                        distone_nodes = []
                        for q in range(0, len(neighbours_nodeall)):
                            distone = []
                            for p in range(0, len(neighbours_nodeall)):
                                dist = distance(
                                    neighbours_nodeall[q], neighbours_nodeall[p]
                                )
                                if dist == 1:
                                    distone.append(neighbours_nodeall[p])
                            distone_nodes.append(distone)
                        numneighbours = []
                        for q in range(0, len(distone_nodes)):
                            numneighbours.append(len(distone_nodes[q]))
                            if (
                                len(distone_nodes[q]) >= 2
                            ):  # Wenn der Abstand zwischen zwei Nachbarn des Nodes 1 beträgt,
                                bo.append(
                                    True
                                )  # dann darf kein weiterer Nachbar des Nodes existieren, der Abstand 1 zu einem der Beiden hat
                            else:
                                bo.append(False)
                        if (
                            0 not in numneighbours
                        ):  # Es muss mind einen Nachbarn des Nodes geben, der nicht direkt neben einem anderen Nachbarn des Nodes liegt
                            bo.append(True)
                        if not any(bo):
                            result[it_x, it_y] = 0
                            bc.append([it_x, it_y])
                    if it_x < binary.shape[0] and it_y < binary.shape[1]:
                        if binary[it_x - 1, it_y - 1] == 1:
                            cross.append(True)
                        if binary[it_x + 1, it_y - 2] == 1:
                            cross.append(True)
                        if binary[it_x, it_y - 1] == 1:
                            cross.append(True)
                        if binary[it_x - 1, it_y] == 1:
                            cross.append(True)
                        if binary[it_x - 2, it_y - 2] == 1:
                            cross.append(True)
                        if binary[it_x - 2, it_y + 1] == 1:
                            cross.append(True)
                        if binary[it_x + 1, it_y + 1] == 1:
                            cross.append(True)
                        if len(cross) == 7:
                            # print('crossing at ', [it_x, it_y])
                            bc.append([it_x, it_y])
                            bc.append([it_x - 1, it_y - 1])
                            bc.append([it_x, it_y - 1])
                            bc.append([it_x - 1, it_y])
                            result[it_x, it_y] = 0
                            result[it_x - 1, it_y - 1] = 0
                            result[it_x, it_y - 1] = 0
                            result[it_x - 1, it_y] = 0

        # plot landmarks
        bcnodes = bc.copy()
        bcnodescoor = []
        allnodes = []
        allnodescoor = []
        endpoints_coor = []
        pltimage = result.copy()
        pltimage = cv2.cvtColor(pltimage, cv2.COLOR_GRAY2RGB)
        for i in range(len(bc)):
            allnodescoor.append([bc[i][1], bc[i][0]])
            allnodes.append([bc[i][0], bc[i][1]])
            bcnodescoor.append([bc[i][1], bc[i][0]])
            cv2.circle(pltimage, (bc[i][1], bc[i][0]), 0, (255, 0, 0), node_thick)
        for i in range(len(endpoints)):
            endpoints_coor.append([endpoints[i][1], endpoints[i][0]])
            cv2.circle(
                pltimage, (endpoints[i][1], endpoints[i][0]), 0, (0, 0, 255), node_thick
            )
            allnodes.append([endpoints[i][0], endpoints[i][1]])
            allnodescoor.append([endpoints[i][1], endpoints[i][0]])

        return bcnodes, endpoints, result  # (
        #     bcnodes,
        #     bcnodescoor,
        #     endpoints,
        #     endpoints_coor,
        #     allnodes,
        #     allnodescoor,
        #     pltimage,
        # )
