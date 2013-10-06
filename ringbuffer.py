# coding=utf-8
"""
Created on Tue Sep 03 13:20:31 2013

This class is a simple ring buffer.

@author: rulas
"""
__author__ = 'rulas'


class RingBuffer(object):
    """
    Convenient ringbuffer

    :param size:
    """

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
        pop an element from the buffer
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

        :param size:
        """
        self._size = size
        self.clear()