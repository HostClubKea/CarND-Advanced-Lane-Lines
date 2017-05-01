import cv2


class PerspectiveTransformer():

    def __init__(self, source, destination, ym_per_pix, xm_per_pix):
        # Define source points
        self.source = source
        # Define corresponding destination points
        self.destination = destination
        # Define perspective transformation matrix
        self.transform_matrix = cv2.getPerspectiveTransform(source, destination)
        # Define perspective unwarp transformation matrix
        self.unwarp_matrix = cv2.getPerspectiveTransform(destination, source)
        # Meters per pixel in y dimension
        self.ym_per_pix = ym_per_pix
        # Meters per pixel in x dimension
        self.xm_per_pix = xm_per_pix

    def transform(self, image):
        # Get image dimensions
        (h, w) = (image.shape[0], image.shape[1])
        return cv2.warpPerspective(image, self.transform_matrix, (w, h))

    def unwarp(self, image):
        # Get image dimensions
        (h, w) = (image.shape[0], image.shape[1])
        return cv2.warpPerspective(image, self.unwarp_matrix, (w, h))
