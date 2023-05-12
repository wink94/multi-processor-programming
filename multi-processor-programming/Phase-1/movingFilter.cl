__kernel void moving_filter(__global float* gray_image, __global float* output_filtered, __global float* filter, int width, int height, int filter_size) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    //printf("get_global_id(0) %d ", x);
    //printf("get_global_id(1) %d ", y);

    if (x < width && y < height) {
        int filter_half_size = filter_size / 2;
        float sum = 0.0f;
        for (int i = -filter_half_size; i <= filter_half_size; ++i) {
            for (int j = -filter_half_size; j <= filter_half_size; ++j) {
                int ix = x + i;
                int iy = y + j;
                //printf("ix = %d ", ix);
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    sum += gray_image[iy * width + ix] * filter[(i + filter_half_size) * filter_size + (j + filter_half_size)];
                }
                /*printf("sum = %f ", sum);*/
            }
        }
        output_filtered[y * width + x] = sum;
    }
}
