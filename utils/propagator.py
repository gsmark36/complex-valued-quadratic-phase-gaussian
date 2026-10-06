# Light propagation and multi-plane loss, adapted from odak (https://github.com/kaanaksit/odak).
# Pure-PyTorch band-limited angular spectrum method (BL-ASM); no custom CUDA kernels are required.

import torch
import logging
from odak.learn.wave import get_propagation_kernel, generate_complex_field, calculate_amplitude
from odak.learn.tools import zero_pad, crop_center, generate_2d_gaussian


def circular_binary_mask(px, py, r, cx=0, cy=0):
    """
    Definition to generate a 2D circular binary mask.

    Parameter
    ---------
    px           : int
                   Pixel count in x.
    py           : int
                   Pixel count in y.
    r            : int
                   Radius of the circle.
    cx           : int, optional
                   Center x-coordinate of the circle.
    cy           : int, optional
                   Center y-coordinate of the circle.

    Returns
    -------
    mask         : torch.tensor
                   Mask [px x py].
    """
    x = torch.linspace(-px / 2., px / 2., px)
    y = torch.linspace(-py / 2., py / 2., py)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = ((X - cx) ** 2 + (Y - cy) ** 2) ** 0.5
    mask = torch.zeros_like(Z)
    mask[Z < r] = 1
    return mask


def custom(field, kernel, aperture = 1.):
    """
    Convolution-based beam propagation with a custom transfer function.

    Parameters
    ----------
    field            : torch.complex
                       Complex field [m x n].
    kernel           : torch.complex
                       Complex transfer function (Fourier domain).
    aperture         : torch.tensor
                       Fourier domain aperture (e.g., pinhole in a typical holographic display).

    Returns
    -------
    result           : torch.complex
                       Final complex field [m x n].
    """
    H = kernel * aperture
    field = zero_pad(field, size=(aperture.size()[-2], aperture.size()[-1]))
    U1 = torch.fft.fftshift(torch.fft.fft2(field)) * aperture
    U2 = H * U1
    result = torch.fft.ifft2(torch.fft.ifftshift(U2))
    return result


class propagator():
    """
    Forward light propagation from the hologram plane to one or more image planes.
    Adapted from `Kavaklı et al., Realistic Defocus Blur for Multiplane Computer-Generated Holography`.
    """
    def __init__(
                 self,
                 resolution = [1920, 1080],
                 wavelengths = [515e-9,],
                 pixel_pitch = 8e-6,
                 number_of_frames = 1,
                 number_of_depth_layers = 1,
                 volume_depth = 1e-2,
                 image_location_offset = 5e-3,
                 propagation_type = 'Bandlimited Angular Spectrum',
                 laser_channel_power = None,
                 aperture = None,
                 aperture_size = None,
                 distances = None,
                 aperture_samples = [20, 20, 5, 5],
                 device = torch.device('cpu')
                ):
        """
        Parameters
        ----------
        resolution              : list
                                  Resolution of the (padded) hologram [H, W].
        wavelengths             : list
                                  Wavelengths of light in meters.
        pixel_pitch             : float
                                  Pixel pitch in meters.
        number_of_frames        : int
                                  Number of hologram frames (one per color primary).
        number_of_depth_layers  : int
                                  Equidistant depth layers within the volume. Overridden by `distances` if given.
        volume_depth            : float
                                  Width of the volume along the propagation direction.
        image_location_offset   : float
                                  Center of the volume along the propagation direction.
        propagation_type        : str
                                  Propagation type, see odak.learn.wave.get_propagation_kernel().
        laser_channel_power     : torch.tensor
                                  Laser channel powers [number_of_frames x number_of_wavelengths].
        aperture                : torch.tensor
                                  Aperture at the Fourier plane.
        aperture_size           : float
                                  Radius of a circular Fourier aperture (used when `aperture` is None).
        distances               : torch.tensor
                                  Propagation distances in meters.
        aperture_samples        : list
                                  Sample counts used by `Impulse Response Fresnel` propagation.
        device                  : torch.device
                                  Device to be used for computation.
        """
        self.device = device
        self.pixel_pitch = pixel_pitch
        self.wavelengths = wavelengths
        self.resolution = resolution
        self.propagation_type = propagation_type
        self.number_of_frames = number_of_frames
        self.number_of_depth_layers = number_of_depth_layers
        self.number_of_channels = len(self.wavelengths)
        self.volume_depth = volume_depth
        self.image_location_offset = image_location_offset
        self.aperture_samples = aperture_samples
        self.aperture_size = aperture_size
        self.init_distances(distances)
        self.init_kernels()
        self.init_channel_power(laser_channel_power)
        self.set_aperture(aperture, aperture_size)

    def init_distances(self, distances):
        """
        Internal function to initialize distances.
        """
        if isinstance(distances, type(None)):
            self.distances = torch.linspace(-self.volume_depth / 2., self.volume_depth / 2., self.number_of_depth_layers) + self.image_location_offset
        else:
            self.distances = torch.as_tensor(distances)
            self.number_of_depth_layers = self.distances.shape[0]

    def init_kernels(self):
        """
        Internal function to allocate the kernel cache.
        """
        self.generated_kernels = torch.zeros(
                                             self.number_of_depth_layers,
                                             self.number_of_channels,
                                             device = self.device
                                            )
        self.kernels = torch.zeros(
                                   self.number_of_depth_layers,
                                   self.number_of_channels,
                                   self.resolution[0] * 2,
                                   self.resolution[1] * 2,
                                   dtype = torch.complex64,
                                   device = self.device
                                  )

    def init_channel_power(self, channel_power):
        """
        Internal function to set the laser channel powers.
        """
        self.channel_power = channel_power
        if isinstance(self.channel_power, type(None)):
            self.channel_power = torch.eye(
                                           self.number_of_frames,
                                           self.number_of_channels,
                                           device = self.device,
                                           requires_grad = False
                                          )

    def set_aperture(self, aperture = None, aperture_size = None):
        """
        Set aperture in the Fourier plane.

        Parameters
        ----------
        aperture        : torch.tensor
                          Aperture at the original resolution of a hologram.
                          If None, a circular aperture of radius `aperture_size` is used.
        aperture_size   : int
                          Radius of the circular aperture on the 2x (zero-padded) Fourier grid.
        """
        if isinstance(aperture, type(None)):
            if isinstance(aperture_size, type(None)):
                aperture_size = max(self.resolution[0], self.resolution[1])
            self.aperture = circular_binary_mask(
                                                 self.resolution[0] * 2,
                                                 self.resolution[1] * 2,
                                                 aperture_size,
                                                ).to(self.device) * 1.
        else:
            self.aperture = zero_pad(aperture).to(self.device) * 1.

    def _ensure_all_kernels_generated(self):
        """
        Pre-computes the transfer functions for all depths and channels (once).
        """
        for depth_id in range(self.number_of_depth_layers):
            distance = self.distances[depth_id]
            for channel_id in range(self.number_of_channels):
                if not self.generated_kernels[depth_id, channel_id]:
                    H = get_propagation_kernel(
                        nu=self.resolution[0] * 2,
                        nv=self.resolution[1] * 2,
                        dx=self.pixel_pitch,
                        wavelength=self.wavelengths[channel_id],
                        distance=distance,
                        device=self.device,
                        propagation_type=self.propagation_type,
                        samples=self.aperture_samples,
                        scale=1
                    )
                    self.kernels[depth_id, channel_id] = H
                    self.generated_kernels[depth_id, channel_id] = True

    def reconstruct(self, hologram_phases, amplitude=None, no_grad=True):
        """
        Reconstruct a given hologram at every depth plane.

        Parameters
        ----------
        hologram_phases : torch.tensor
                         Hologram phases [ch x m x n].
        amplitude       : torch.tensor
                         Amplitude profiles for each color primary [ch x m x n].
        no_grad         : bool
                         If set True, uses torch.no_grad in reconstruction.

        Returns
        -------
        reconstructions : torch.tensor
                         Reconstructed intensities [frames x depths x channels x m x n].
        """
        if no_grad:
            with torch.no_grad():
                return self._reconstruct_impl(hologram_phases, amplitude)
        else:
            return self._reconstruct_impl(hologram_phases, amplitude)

    def _reconstruct_impl(self, hologram_phases, amplitude=None):
        if len(hologram_phases.shape) > 3:
            hologram_phases = hologram_phases.squeeze(0)

        if hologram_phases.shape[0] != self.number_of_frames:
            logging.warning('Provided hologram frame count is {} but the configured number of frames is {}.'.format(
                hologram_phases.shape[0], self.number_of_frames))

        reconstructions = torch.zeros(
            self.number_of_frames,
            self.number_of_depth_layers,
            self.number_of_channels,
            self.resolution[0],
            self.resolution[1],
            dtype=torch.float32,
            device=self.device
        )

        if isinstance(amplitude, type(None)):
            amplitude = torch.ones(
                self.number_of_channels,
                self.resolution[0],
                self.resolution[1],
                device=self.device
            )

        self._ensure_all_kernels_generated()

        for frame_id in range(self.number_of_frames):
            phase = hologram_phases[frame_id]
            for depth_id in range(self.number_of_depth_layers):
                for channel_id in range(self.number_of_channels):
                    laser_power = self.channel_power[frame_id][channel_id]
                    # A frame that does not light this channel contributes exactly zero; skip it.
                    if laser_power == 0:
                        continue
                    hologram = generate_complex_field(laser_power * amplitude[channel_id], phase)
                    H = self.kernels[depth_id, channel_id]
                    field_scale_padded = zero_pad(hologram)
                    output_field_padded = custom(field_scale_padded, H, aperture=self.aperture)
                    output_field = crop_center(output_field_padded)
                    reconstructions[frame_id, depth_id, channel_id] = calculate_amplitude(output_field) ** 2
        return reconstructions


class multiplane_loss_odak():
    """
    Loss function for computing loss in multiplanar images. Unlike previous methods, this loss function accounts for defocused parts of an image.
    """

    def __init__(self, target_image, target_depth, blur_ratio = 0.25,
                 target_blur_size = 10, number_of_planes = 4, weights = [1., 2.1, 0.6],
                 multiplier = 1., scheme = 'defocus', reduction = 'mean', split_ratio = 1.0, device = torch.device('cpu')):
        """
        Parameters
        ----------
        target_image      : torch.tensor
                            Target image [C x m x n].
        target_depth      : torch.tensor
                            Monochrome target depth in [0, 1], same resolution as target_image.
        target_blur_size  : int
                            Maximum target blur size.
        blur_ratio        : float
                            Blur ratio.
        number_of_planes  : int
                            Number of planes.
        weights           : list
                            Weights of the loss function.
        multiplier        : float
                            Multiplier to multiply with targets.
        scheme            : str
                            The type of the loss, `naive` without defocus or `defocus` with defocus.
        reduction         : str
                            Reduction can either be 'mean', 'none' or 'sum'.
        split_ratio       : float
                            Exponent applied to the depth before splitting into two planes
                            (<1 biases pixels to the front plane, >1 to the rear plane).
        device            : torch.device
                            Device to be used.
        """
        self.device = device
        self.target_image     = target_image.float().to(self.device)
        self.target_depth     = target_depth.float().to(self.device)
        self.target_blur_size = target_blur_size
        if self.target_blur_size % 2 == 0:
            self.target_blur_size += 1
        self.number_of_planes = number_of_planes
        self.multiplier       = multiplier
        self.weights          = weights
        self.reduction        = reduction
        self.blur_ratio       = blur_ratio
        self.split_ratio      = split_ratio
        self.set_targets()
        if scheme == 'defocus':
            self.add_defocus_blur()
        self.loss_function = torch.nn.MSELoss(reduction = self.reduction)

    def get_targets(self):
        """
        Returns
        -------
        targets           : torch.tensor
                            A copy of the per-plane targets.
        masks             : torch.tensor
                            A copy of the per-plane masks.
        target_depth      : torch.tensor
                            A copy of the normalized quantized depth map.
        """
        divider = self.number_of_planes - 1
        if divider == 0:
            divider = 1
        return self.targets.detach().clone(), self.masks.detach().clone(), self.target_depth.detach().clone() / divider

    def set_targets(self):
        """
        Internal function for slicing the depth into planes without considering defocus.
        """
        normalized_depth = self.target_depth
        if self.number_of_planes == 2:
            ratio = self.split_ratio
        else:
            ratio = 1.0
        biased_depth = torch.pow(normalized_depth, ratio)
        biased_depth = biased_depth * (self.number_of_planes - 1)
        self.target_depth = torch.round(biased_depth, decimals=0)

        self.targets      = torch.zeros(
                                        self.number_of_planes,
                                        self.target_image.shape[0],
                                        self.target_image.shape[1],
                                        self.target_image.shape[2],
                                        requires_grad = False,
                                        device = self.device
                                       )
        self.masks        = torch.zeros_like(self.targets)
        for i in range(self.number_of_planes):
            for ch in range(self.target_image.shape[0]):
                mask_zeros = torch.zeros_like(self.target_image[ch], dtype = torch.int)
                mask_ones = torch.ones_like(self.target_image[ch], dtype = torch.int)
                mask = torch.where(self.target_depth == i, mask_ones, mask_zeros)
                new_target = self.target_image[ch] * mask
                self.targets[i, ch] = new_target.squeeze(0).squeeze(0)
                self.masks[i, ch] = mask.detach().clone()

    def add_defocus_blur(self):
        """
        Internal function for adding defocus blur to the multiplane targets.
        """
        kernel_length = [self.target_blur_size, self.target_blur_size ]
        for ch in range(self.target_image.shape[0]):
            targets_cache = self.targets[:, ch].detach().clone()
            target = torch.sum(targets_cache, axis = 0)
            for i in range(self.number_of_planes):
                defocus = torch.zeros_like(targets_cache[i])
                for j in range(self.number_of_planes):
                    nsigma = [int(abs(i - j) * self.blur_ratio), int(abs(i -j) * self.blur_ratio)]
                    if torch.sum(targets_cache[j]) > 0:
                        if i == j:
                            nsigma = [0., 0.]
                        kernel = generate_2d_gaussian(kernel_length, nsigma).to(self.device)
                        kernel = kernel / torch.sum(kernel)
                        kernel = kernel.unsqueeze(0).unsqueeze(0)
                        target_current = target.detach().clone().unsqueeze(0).unsqueeze(0)
                        defocus_plane = torch.nn.functional.conv2d(target_current, kernel, padding = 'same')
                        defocus_plane = defocus_plane.view(defocus_plane.shape[-2], defocus_plane.shape[-1])
                        defocus = defocus + defocus_plane * torch.abs(self.masks[j, ch])
                self.targets[i, ch] = defocus
        self.targets = self.targets.detach().clone() * self.multiplier

    def __call__(self, image, target, plane_id = None, inject_noise = False, noise_ratio = 1e-3):
        """
        Calculates the multiplane loss against a given target.

        Parameters
        ----------
        image         : torch.tensor
                        Image to compare with a target [C x m x n].
        target        : torch.tensor
                        Target image for comparison [C x m x n].
        plane_id      : int
                        Number of the plane under test.
        inject_noise  : bool
                        When True, noise is added on the targets at the given `noise_ratio`.
        noise_ratio   : float
                        Noise ratio.

        Returns
        -------
        loss          : torch.tensor
                        Computed loss.
        """
        l2 = self.weights[0] * self.loss_function(image, target)
        if isinstance(plane_id, type(None)):
            mask = self.masks
        else:
            mask= self.masks[plane_id, :]
        if inject_noise:
            target = target + torch.randn_like(target) * noise_ratio * (target.max() - target.min())
        l2_mask = self.weights[1] * self.loss_function(image * mask, target * mask)
        l2_cor = self.weights[2] * self.loss_function(image * target, target * target)
        loss = l2 + l2_mask + l2_cor
        return loss
