import vtk
from vtk.util import numpy_support
import numpy as np
import itk
import copy


imageType = itk.Image[itk.F, 3]

def itk_to_vtk(itkImage):

    resample_factor = 10



    array = itk.GetArrayFromImage(itkImage)

    #Downsample test
    spacing = itkImage.GetSpacing()
    spacing[2] *= resample_factor
    array =array[::resample_factor, :, :]
    dims = (array.shape[1], array.shape[2], array.shape[0])



    downsampled_image = itk.GetImageFromArray(array)
    print(downsampled_image)
    
    vtk_array = numpy_support.numpy_to_vtk(num_array=array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    

    print(dims, spacing)

    #MN
    vtkImage = vtk.vtkImageData()
    vtkImage.SetDimensions(dims)
    vtkImage.SetSpacing(spacing)
    vtkImage.GetPointData().SetScalars(vtk_array)

    return vtkImage



readerType = itk.ImageFileReader[imageType]

reader = readerType.New()
reader.SetFileName("./volume-5.nii")
reader.Update()

itkImage = reader.GetOutput()
vtkImage = itk_to_vtk(itkImage)
scalar_range = vtkImage.GetScalarRange()

mapper = vtk.vtkSmartVolumeMapper()
mapper.SetInputData(vtkImage)
mapper.SetBlendModeToMaximumIntensity()

volume_property = vtk.vtkVolumeProperty()
volume_property.ShadeOff()
#volume_property.SetInterpolationTypeToMaximumIntensity()

opac_func = vtk.vtkPiecewiseFunction()
opac_func.AddPoint(scalar_range[0], 0.0)
opac_func.AddPoint(scalar_range[1], 1.0)
volume_property.SetScalarOpacity(opac_func)

color_func = vtk.vtkColorTransferFunction()
color_func.AddRGBPoint(scalar_range[0], 0.0, 0.0, 0.0)
color_func.AddRGBPoint(scalar_range[1], 1.0, 1.0, 1.0)
volume_property.SetColor(color_func)

volume = vtk.vtkVolume()
volume.SetMapper(mapper)
volume.SetProperty(volume_property)


ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
# Add the actors to the renderer, set the background and size
ren.AddViewProp(volume)


ren.SetBackground(0.0, 0.0, 0.01)
renWin.SetSize(500, 500)

# This allows the interactor to initalize itself. It has to be
# called before an event loop.
iren.Initialize()

# We'll zoom in a little by accessing the camera and invoking a "Zoom"
# method on it.
ren.ResetCamera()
ren.GetActiveCamera().Zoom(1.5)
renWin.Render()

# Start the event loop.
iren.Start()