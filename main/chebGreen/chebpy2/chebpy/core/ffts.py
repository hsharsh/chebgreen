# """Import fftw via pyfftw if the user has it installed,
# otherwise default to fftpack via numpy"""

# from .importing import import_optional


# # import the requested FFT module
# _fft = import_optional("pyfftw.interfaces.numpy_fft", "PYFFTW", fallback="numpy.fft")

# import the requested FFT module
import pyfftw.interfaces.numpy_fft as _fft

# assign the interfaces for import from other modules
fft, ifft = _fft.fft, _fft.ifft
