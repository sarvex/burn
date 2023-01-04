#[burn_tensor_testgen::testgen(module_conv2d)]
mod tests {
    use super::*;
    use burn_tensor::module::conv2d;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_conv2d_simple_1() {
        let x = TestTensor::from_floats([
            [
                [
                    [0.4745, 0.6486, 0.1103, 0.4676],
                    [0.1310, 0.3708, 0.0497, 0.7468],
                    [0.9011, 0.9896, 0.3942, 0.1292],
                ],
                [
                    [0.6909, 0.2882, 0.8625, 0.8999],
                    [0.3282, 0.4600, 0.6506, 0.5551],
                    [0.8251, 0.4340, 0.6002, 0.8908],
                ],
                [
                    [0.0323, 0.8586, 0.0223, 0.9322],
                    [0.8614, 0.7229, 0.1213, 0.2052],
                    [0.1388, 0.2891, 0.6902, 0.7284],
                ],
            ],
            [
                [
                    [0.2508, 0.6567, 0.6622, 0.1617],
                    [0.7283, 0.2524, 0.1930, 0.3182],
                    [0.7414, 0.7586, 0.4549, 0.5657],
                ],
                [
                    [0.3260, 0.9229, 0.9120, 0.0860],
                    [0.7636, 0.8822, 0.0675, 0.0820],
                    [0.4536, 0.7230, 0.3965, 0.6649],
                ],
                [
                    [0.3959, 0.5494, 0.1427, 0.2821],
                    [0.2458, 0.5083, 0.1261, 0.5584],
                    [0.6385, 0.9546, 0.9800, 0.3996],
                ],
            ],
        ]);
        let y = TestTensor::from_floats([
            [
                [
                    [0.2897, 0.2466, 0.4861, 0.1689],
                    [0.3304, 0.3087, 0.1674, -0.0117],
                    [0.0317, 0.0010, 0.1062, 0.1829],
                ],
                [
                    [-0.2228, -0.4840, -0.1837, -0.4934],
                    [-0.2426, -0.5943, 0.0093, -0.5536],
                    [-0.3359, -0.2840, -0.2798, -0.2942],
                ],
                [
                    [-0.1541, 0.0406, -0.2952, -0.1628],
                    [-0.2383, -0.2156, -0.1318, -0.3655],
                    [-0.3832, -0.3135, -0.2512, -0.4277],
                ],
            ],
            [
                [
                    [0.5086, 0.2759, 0.2826, -0.1429],
                    [0.3092, 0.1823, 0.1705, 0.1725],
                    [0.1312, 0.3331, 0.0671, 0.0812],
                ],
                [
                    [-0.3082, -0.4687, -0.2607, -0.0913],
                    [-0.4234, -0.2464, -0.1810, -0.3279],
                    [-0.4534, -0.4499, -0.2359, -0.1634],
                ],
                [
                    [0.0766, -0.2484, -0.4462, 0.0347],
                    [-0.1236, -0.2826, -0.1969, 0.0196],
                    [-0.4907, -0.5210, -0.2815, -0.4017],
                ],
            ],
        ]);
        let weights = TestTensor::from_floats([
            [
                [
                    [1.2395e-01, 3.5688e-02, -1.8216e-01],
                    [1.2230e-01, 3.6156e-02, 1.0928e-01],
                    [-1.6161e-01, 1.4208e-01, 1.0860e-01],
                ],
                [
                    [1.9558e-02, 1.2546e-01, -9.3963e-02],
                    [-1.8986e-01, 1.6779e-01, 8.3757e-02],
                    [7.8266e-02, 1.5001e-02, 1.6319e-01],
                ],
                [
                    [-5.1963e-02, -1.1949e-01, -1.1350e-01],
                    [9.3419e-03, 5.3613e-02, -2.0891e-02],
                    [7.8037e-02, -1.4770e-01, 1.4043e-01],
                ],
            ],
            [
                [
                    [1.7307e-02, 3.8177e-03, 7.4619e-02],
                    [1.0092e-01, -1.1778e-01, 1.0228e-01],
                    [3.8074e-02, -3.5568e-02, 5.0294e-02],
                ],
                [
                    [-1.3636e-01, -3.3311e-02, -3.9072e-02],
                    [-8.1162e-03, -1.2371e-01, -1.8651e-01],
                    [-1.7376e-01, 3.6007e-02, -1.5975e-02],
                ],
                [
                    [1.6726e-01, -1.0763e-01, 4.3256e-02],
                    [1.0587e-01, -1.7313e-01, -5.6606e-02],
                    [2.2199e-02, -5.7266e-02, 5.0295e-02],
                ],
            ],
            [
                [
                    [1.0021e-01, -1.4624e-01, -4.9954e-02],
                    [1.0525e-02, -1.4658e-01, -1.1044e-04],
                    [-3.3403e-02, 1.7280e-01, 2.2788e-03],
                ],
                [
                    [7.7067e-02, -1.6466e-01, 8.8795e-02],
                    [2.9386e-02, -9.2502e-02, 9.8240e-02],
                    [-1.2840e-01, 9.6244e-03, 1.0160e-01],
                ],
                [
                    [3.1879e-02, -4.7822e-02, -4.6569e-02],
                    [-1.4448e-01, -7.0079e-02, -1.2398e-01],
                    [-1.2211e-02, 1.4308e-01, -1.2552e-01],
                ],
            ],
        ]);
        let bias = TestTensor::from_floats([-0.0354, -0.0454, -0.0460]);

        let output = conv2d(&x, &weights, Some(&bias), [1, 1], [1, 1]);

        y.to_data().assert_approx_eq(&output.into_data(), 3);
    }
}