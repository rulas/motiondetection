__author__ = 'lrvillan'


class RingBuffer():
    def __init__(self, size):
        self._size = size
        self.clear()

    def push(self, element):
        """
        inserts an element into the buffer
        :param element: element to be inserted
        """
        self._buffer.append(element)
        self._buffer.pop(0)

    def pop(self):
        """
        pop an element frmom the buffer
        """
        pass

    @property
    def buffer(self):
        """
        return the list of elements in the buffer

        :return: list of elements
        """
        return self._buffer

    def clear(self):
        """
        clears the buffer

        """
        self._buffer = [0] * self._size

    def resize(self, size):
        """
        clears the buffer and create a buffer with the given size

        :param new_size: new size of buffer
        """
        self._size = size
        self.clear()