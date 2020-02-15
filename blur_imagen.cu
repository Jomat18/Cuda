#include <iostream>
#include <vector>

#include <cuda.h>
#include <vector_types.h>
#include "bitmap_image.hpp"
#define BLUR_SIZE 7

using namespace std;

__global__ void blurKernel(uchar3 *in, uchar3 *out, int w, int h)
{
        int Col = blockIdx.x*blockDim.x + threadIdx.x;
        int Row = blockIdx.y*blockDim.y + threadIdx.y;

        if(Col<w && Row<h)
        {
            int pixVal1 = 0;
           // int pixVal2 = 0;
	   // int pixVal3 = 0;
            int	pixels1 = 0;
           // int pixels2 = 0;
           // int pixels3 = 0;

            for(int blurRow=-BLUR_SIZE; blurRow<BLUR_SIZE+1;++blurRow){
                for(int blurCol=-BLUR_SIZE; blurCol<BLUR_SIZE+1;++blurCol)
                {
                    int curRow = Row + blurRow;
                    int curCol = Col + blurCol;

                    if(curRow>-1 && curRow<h && curCol>-1 && curCol<w){
                        pixVal1+=static_cast<int>(in[curRow*w + curCol].x);
                        pixels1++;
			pixVal1+=static_cast<int>(in[curRow*w + curCol].y);
			pixels1++;
			pixVal1+=static_cast<int>(in[curRow*w + curCol].z);
			pixels1++;

                    }
                }

            }

            out[Row*w+Col].x= static_cast<unsigned char>(pixVal1/pixels1); 
	    out[Row*w+Col].y= static_cast<unsigned char>(pixVal1/pixels1);
	    out[Row*w+Col].z= static_cast<unsigned char>(pixVal1/pixels1);	

        }
}


int main()
{
    bitmap_image bmp("lenna.bmp");

    if(!bmp)
    {
        cerr << "Image not found" << endl;
        exit(1);
    }

    int height = bmp.height();
    int width = bmp.width();
    
    cout << "image dimensions" << endl;
    cout << "height " << height << " width " << width << endl;

    //Transform image into vector of doubles
    vector<uchar3> input_image;
    rgb_t color;
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            bmp.get_pixel(x, y, color);
            input_image.push_back( {color.red, color.green, color.blue} );
        }
    }

    vector<uchar3> output_image(input_image.size());

    uchar3 *d_in, *d_out;
    int img_size = (input_image.size() * sizeof(char) * 3);
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);

    cudaMemcpy(d_in, input_image.data(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, input_image.data(), img_size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width / 16), ceil(height / 16), 1);
    dim3 dimBlock(16, 16, 1);

    blurKernel<<< dimGrid , dimBlock >>> (d_in, d_out, width, height);

    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    
    
    //Establecer píxeles actualizados
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            int pos = x * width + y;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }
    
    bmp.save_image("./blur.bmp");

    cudaFree(d_in);
    cudaFree(d_out);
}
