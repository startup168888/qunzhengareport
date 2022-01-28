## Copyright (c) 2009-2010, Florian Finkernagel. All rights reserved.

## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:

##     * Redistributions of source code must retain the above copyright
##       notice, this list of conditions and the following disclaimer.

##     * Redistributions in binary form must reproduce the above
##       copyright notice, this list of conditions and the following
##       disclaimer in the documentation and/or other materials provided
##       with the distribution.

##     * Neither the name of the Florian Finkernagel nor the names of its
##       contributors may be used to endorse or promote products derived
##       from this software without specific prior written permission.

## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy
import re

def _find_levels(values):
    order = []
    existance = {}
    for v in values:
        if not v in existance:
            existance[v] = True
            order.append(v)
    return numpy.array(order)


class Factor(numpy.ndarray):
    """A Factor is a numpy array constrained to a few values
    that each have a unique label."""

    def __new__(cls, values, levels = None, transform_data = True):
        if levels is None:
            levels = _find_levels(values)
        else:
            levels = list(levels)
            if len(set(levels)) != len(levels):
                raise ValueError("Duplicate values in levels are not allowed.")
        level_lookup = {}
        for ii, ll in enumerate(levels):
            level_lookup[ll] = ii
        if transform_data:
            try:
                obj = numpy.array([level_lookup[x] for x in values], dtype=numpy.int32) #  I want the list comprehension to leak the variable
            except KeyError:
                for x in values:
                    if not x in level_lookup:
                        raise ValueError("Missing level value (but existing data point): %s (levels were %s)" % (x, level_lookup.keys()))
                raise ValueError("Missing level value, but apperantly in level_lookup? Should not happen")
                
        else:
                obj = numpy.array(values, dtype=numpy.int32)


        obj = obj.view(cls)
        obj.levels = list(levels)
        obj.level_lookup = level_lookup
        return obj

    def __array_finalize__(self,obj):
        self.levels = getattr(obj, 'levels', None)
        self.level_lookup = getattr(obj, 'level_lookup', None)

    def map_level(self, level):
        """Map a level to the appropriate stored value"""
        return self.level_lookup[level]

    def map_value(self, value):
        """Map a value to a level (label)"""
        return self.levels[value]

    def __str__(self):
        s = numpy.ndarray.__str__(self)
        for level, number in self.level_lookup.items():
            s = re.sub('([\\[ ])%i([ \\]])' % number, '\\1%s\\2' % level, s)
        level_str = " < ".join(self.levels)

        return "Factor array:\n" + s + '\nLevels: ' + level_str

    def as_levels(self):
        res = []
        for v in self:
            res.append(self.levels[v])
        return numpy.array(res)

    def __eq__(self, other):
        if isinstance(other, str) and other in self.levels:
            return self == self.levels.index(other)
        else:
            return numpy.ndarray.__eq__(self, other)
