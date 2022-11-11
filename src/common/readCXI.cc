#include "opencv2/opencv.hpp" 
#include "hdf5.h"
#include "common.h"
using namespace cv;


#define RANK         2
#define RANK_OUT     2

Mat readCXI (const char* filename, Mat **mask)
{
    int noiseLevel = 550;
    hid_t       file, dataset, entry, imagehd, maskhd;         /* handles */
    hid_t       datatype, dataspace;   
    hid_t       memspace; 
    H5T_class_t classm;                 /* datatype class */
    H5T_order_t order;                 /* data order */
    size_t      size;                  /*
				        * size of the data element	       
				        * stored in file
				        */
    hsize_t     dimsm[2];              /* memory space dimensions */
    hsize_t     dims_out[2];           /* dataset dimensions */      
    herr_t      status;                             

   
    hsize_t      count[2];              /* size of the hyperslab in the file */
    hsize_t      offset[2];             /* hyperslab offset in the file */
    hsize_t      count_out[2];          /* size of the hyperslab in memory */
    hsize_t      offset_out[2];         /* hyperslab offset in memory */
    int          i, j, k, status_n, rank;

    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    entry = H5Gopen(file, "entry_1", H5P_DEFAULT);
    imagehd = H5Gopen(entry, "image_1", H5P_DEFAULT);
    dataset = H5Dopen(imagehd, "data",H5P_DEFAULT);
    maskhd = H5Dopen(imagehd, "mask",H5P_DEFAULT);

    datatype  = H5Dget_type(dataset);     /* datatype handle */ 
    classm     = H5Tget_class(datatype);
    if (classm == H5T_INTEGER) printf("Data set has INTEGER type \n");
    else if(classm == H5T_COMPOUND) {
	    printf("CMPOUND DATATYPE {\n");
	    printf(" %ld bytes\n",H5Tget_size(datatype));
	    printf(" %d members\n",H5Tget_nmembers(datatype));
    }

    order     = H5Tget_order(datatype);
    if (order == H5T_ORDER_LE) printf("Little endian order \n");

    size  = H5Tget_size(datatype);
    printf(" Data size is %ld \n", size);

    dataspace = H5Dget_space(dataset);    /* dataspace handle */
    rank      = H5Sget_simple_extent_ndims(dataspace);
    status_n  = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
    printf("rank %d, dimensions %lu x %lu \n", rank,
	   (unsigned long)(dims_out[0]), (unsigned long)(dims_out[1]));

    Mat image(dims_out[0],dims_out[1],CV_32FC2);
    Mat imageint(dims_out[0],dims_out[1],CV_8UC1);
    Mat imagelog(dims_out[0],dims_out[1],CV_8UC1);
    if(mask) *mask = new Mat(dims_out[0],dims_out[1],CV_8UC1);
    int *maskdata = (int*)malloc(dims_out[0]*dims_out[1]*sizeof(int));
    memspace = H5Screate_simple(RANK_OUT,dims_out,NULL);   
    hid_t complex_id = H5Tcreate(H5T_COMPOUND,sizeof(float)*2);
    H5Tinsert(complex_id,"r",0,H5T_NATIVE_FLOAT);
    H5Tinsert(complex_id,"i",sizeof(float),H5T_NATIVE_FLOAT);
    status = H5Dread(dataset, complex_id, memspace, dataspace,
		     H5P_DEFAULT, image.data);
    status = H5Dread(maskhd, H5T_STD_I32LE, memspace, dataspace,
		     H5P_DEFAULT, maskdata);
    Real noiseScale = (rcolor-1)/(rcolor-noiseLevel-1);
    for(int i = 0 ; i < image.total(); i++){
	    auto &datai = ((complex<float>*)image.data)[i];
	    datai-=noiseLevel;
	    datai *= noiseScale;
	    ((char*)imageint.data)[i] = (int)(datai.real()/256);
	    ((char*)imagelog.data)[i] = datai.real() > 1?(int)(log(datai.real())*16):0;
	    if(mask) ((char*)(*mask)->data)[i] = maskdata[i]*255;
    }
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Sclose(memspace);
    H5Fclose(file);
    imwrite("cxi.png",imageint);
    if(mask)imwrite("cximask.png",**mask);
    imwrite("cxilog.png",imagelog);

    return image;
}     

