#include <vtkNew.h>
#include <vtkVolumeProperty.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkVolume.h>
#include "vtkGPUVolumeRayCastMapper.h"
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkNamedColors.h>
#include <vtkAutoInit.h>
#include "format.h"
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingVolumeOpenGL2)

int writeVTI(const char* filename)
{
  vtkNew<vtkNamedColors> colors;

  // Parse command line arguments
  Real opacityWindow = 1.;
  Real opacityLevel = 0.5;
  bool createFile = 1;

  if(createFile){
    vtkNew<vtkImageData> imageData;
    imageData->SetDimensions(30, 40, 50);
    imageData->AllocateScalars(VTK_TYPE, 2);

    int* dims = imageData->GetDimensions();
    //processComplex(data);

    // Fill every entry of the image data with "2.0"
    for (int z = 0; z < dims[2]; z++)
    {
      for (int y = 0; y < dims[1]; y++)
      {
        for (int x = 0; x < dims[0]; x++)
        {
          Real* pixel = static_cast<Real*>(imageData->GetScalarPointer(x, y, z));
          pixel[0] =  255.*exp(-pow(hypot(x-15, y-20)/10,2)); //control color
          pixel[1] =  255.*exp(-pow(hypot(z-25, y-20)/10,2)); //control opacity
        }
      }
    }
    vtkNew<vtkXMLImageDataWriter> writer;
    writer->SetFileName(filename);
    writer->SetInputData(imageData);
    writer->Write();
  }
  // Read the file (to test that it was written correctly)
  vtkNew<vtkXMLImageDataReader> reader;
  reader->SetFileName(filename);
  reader->Update();

  vtkNew<vtkGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(reader->GetOutputPort());
  //mapper->SetInputData(imageData);

  vtkNew<vtkColorTransferFunction> colorFun;
  vtkNew<vtkPiecewiseFunction> opacityFun;
  // Create the property and attach the transfer functions
  vtkNew<vtkVolumeProperty> property;
  property->SetIndependentComponents(false);
  property->SetColor(colorFun);
  property->SetScalarOpacity(opacityFun);
  property->SetInterpolationTypeToLinear();
  colorFun->AddRGBPoint(0.0, 0.0, 0.0, 0.0);
  colorFun->AddRGBPoint(64.0, 1.0, 0.0, 0.0);
  colorFun->AddRGBPoint(128.0, 0.0, 0.0, 1.0);
  colorFun->AddRGBPoint(192.0, 0.0, 1.0, 0.0);
  colorFun->AddRGBPoint(255.0, 0.0, 0.2, 0.0);
  opacityFun->AddPoint(100, 0.0);
  opacityFun->AddPoint(120., 1);
  mapper->SetBlendModeToComposite();
  property->ShadeOn();
  //property->SetAmbient(1.);
  //property->SetDiffuse(0.0);
  //property->SetSpecular(0.);

  // connect up the volume to the property and the mapper
  vtkNew<vtkVolume> volume;
  volume->SetProperty(property);
  volume->SetMapper(mapper);

  // Setup rendering
  vtkNew<vtkRenderer> renderer;
  renderer->AddVolume(volume);
  renderer->SetBackground(colors->GetColor3d("AliceBlue").GetData());
  renderer->ResetCamera();

  vtkNew<vtkRenderWindow> renderWindow;
  renderWindow->AddRenderer(renderer);
  renderWindow->SetWindowName("WriteVTI");


  vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;

  renderWindowInteractor->SetRenderWindow(renderWindow);
  renderWindow->Render();
  renderWindowInteractor->Initialize();
  renderWindowInteractor->Start();

  return EXIT_SUCCESS;
}
