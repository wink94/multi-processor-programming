// Phase-3.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "lodepng.h"


using namespace std;


vector<unsigned char> readImage(const std::string& filename) {
    std::vector<unsigned char> image;
    unsigned width, height;
    unsigned error = lodepng::decode(image, width, height, filename);

    if (error) {
        std::cout << "Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }

    return image;
}

void resizeImage(const unsigned char* original_image, unsigned char* resized_image, unsigned original_width, unsigned original_height) {
	unsigned resized_width = original_width / 4;
	unsigned resized_height = original_height / 4;

	for (unsigned y = 0; y < resized_height; ++y) {
		for (unsigned x = 0; x < resized_width; ++x) {
			unsigned original_index = (y * 4 * original_width + x * 4) * 4;
			unsigned resized_index = (y * resized_width + x) * 4;

			for (int k = 0; k < 4; ++k) {
				resized_image[resized_index + k] = original_image[original_index + k];
			}
		}
	}
}

int main()
{
    std::cout << "Hello World!\n";
}

