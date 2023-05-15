Usage
=====

Using the camera
----------------

To use the camera you can use the ``oakutils.Camera`` class:

.. code-block:: python

   from oakutils import Camera

   camera = Camera()
   camera.start()
   frame = camera.depth
   camera.stop()
