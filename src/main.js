import * as tf from '@tensorflow/tfjs';
import { Matrix } from 'ml-matrix';
import { ConfusionMatrix } from 'ml-confusion-matrix';
import Plotly from 'plotly.js-dist';

class Graph {
    constructor(layout = 'coco_cut', strategy = 'uniform', maxHop = 1, dilation = 1) {
        this.maxHop = maxHop;
        this.dilation = dilation;
        this.getEdge(layout);
        this.hopDis = this.getHopDistance(this.numNode, this.edge, maxHop);
        this.getAdjacency(strategy);
    }

    getEdge(layout) {
        if (layout === 'coco_cut') {
            this.numNode = 30;
            const selfLink = Array.from({length: this.numNode}, (_, i) => [i, i]);
            const neighborLink = [
                [6, 4], [4, 2], [2, 13], [13, 1], [5, 3], [3, 1], [12, 10],
                [10, 8], [8, 13], [11, 9], [9, 7], [7, 13], [13, 0], [20, 18],
                [18, 16], [16, 27], [27, 15], [19, 17], [17, 15], [26, 24],
                [24, 22], [22, 27], [25, 23], [23, 21], [21, 27], [27, 14]
            ];
            this.edge = [...selfLink, ...neighborLink];
            this.center = 13;
        }
    }

    getHopDistance(numNode, edge, maxHop = 1) {
        const A = new Matrix(numNode, numNode);
        edge.forEach(([i, j]) => {
            A.set(j, i, 1);
            A.set(i, j, 1);
        });

        const hopDis = Matrix.ones(numNode, numNode).mul(Infinity);
        const transferMat = Array.from({length: maxHop + 1}, 
            (_, d) => Matrix.pow(A, d));

        for (let d = maxHop; d >= 0; d--) {
            const arriveMat = transferMat[d].clone();
            arriveMat.apply(x => x > 0 ? d : x);
            hopDis.add(arriveMat);
        }

        return hopDis;
    }

    getAdjacency(strategy) {
        const validHop = Array.from(
            {length: Math.floor(this.maxHop / this.dilation) + 1},
            (_, i) => i * this.dilation
        );

        const adjacency = new Matrix(this.numNode, this.numNode);
        validHop.forEach(hop => {
            this.hopDis.apply((x, i, j) => {
                if (x === hop) adjacency.set(i, j, 1);
            });
        });

        const normalizedAdjacency = this.normalizeDigraph(adjacency);

        if (strategy === 'uniform') {
            this.A = tf.tensor3d([normalizedAdjacency.to2DArray()]);
        } else if (strategy === 'distance') {
            const A = validHop.map(hop => {
                const hopMatrix = normalizedAdjacency.clone();
                hopMatrix.apply((x, i, j) => 
                    this.hopDis.get(i, j) === hop ? x : 0
                );
                return hopMatrix.to2DArray();
            });
            this.A = tf.tensor3d(A);
        } else if (strategy === 'spatial') {
            const A = [];
            validHop.forEach(hop => {
                const aRoot = new Matrix(this.numNode, this.numNode);
                const aClose = new Matrix(this.numNode, this.numNode);
                const aFurther = new Matrix(this.numNode, this.numNode);

                for (let i = 0; i < this.numNode; i++) {
                    for (let j = 0; j < this.numNode; j++) {
                        if (this.hopDis.get(j, i) === hop) {
                            if (this.hopDis.get(j, this.center) === this.hopDis.get(i, this.center)) {
                                aRoot.set(j, i, normalizedAdjacency.get(j, i));
                            } else if (this.hopDis.get(j, this.center) > this.hopDis.get(i, this.center)) {
                                aClose.set(j, i, normalizedAdjacency.get(j, i));
                            } else {
                                aFurther.set(j, i, normalizedAdjacency.get(j, i));
                            }
                        }
                    }
                }

                if (hop === 0) {
                    A.push(aRoot.to2DArray());
                } else {
                    A.push(aRoot.add(aClose).to2DArray());
                    A.push(aFurther.to2DArray());
                }
            });
            this.A = tf.tensor3d(A);
        } else {
            throw new Error("This strategy is not supported!");
        }
    }

    normalizeDigraph(A) {
        const Dl = A.sum('row');
        const Dn = Matrix.zeros(A.rows, A.columns);
        
        for (let i = 0; i < A.rows; i++) {
            if (Dl[i] > 0) {
                Dn.set(i, i, 1 / Dl[i]);
            }
        }

        return A.mmul(Dn);
    }
}

// Initialize the application
async function init() {
    await tf.ready();
    console.log('TensorFlow.js is ready');
    
    // Create graph instance
    const graph = new Graph();
    console.log('Graph initialized');
}

init().catch(console.error);