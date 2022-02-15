## Copyright (c) 2001-2006, Andrew Straw. All rights reserved.
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

##     * Neither the name of the Andrew Straw nor the names of its
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


from __future__ import print_function
import io
import numpy
try:
    import cPickle as pickle
except ImportError:
    import pickle
import zlib
from .factors import Factor, _find_levels
import sys
import itertools
import six
try:
    from builtins import range
except ImportError:
    range = xrange  # python 2 fallback
import functools
try:
    unicode
except NameError:
    unicode = str
try:
    izip = itertools.izip
except:
    izip = zip


class DataFrame(object):
    """An implemention of an almost R like DataFrame object.

    Usage::

        u = DataFrame( { "Field1": [1, 2, 3],
                        "Field2": ['abc', 'def', 'hgi']},
                        optional:
                         ['Field1', 'Field2']
                         ["rowOne", "rowTwo", "thirdRow"])

    A DataFrame is basically a table with rows and columns.
    Columns are named, rows are numbered (but can be named)
    and can be easily selected and calculated upon.
    Internally, columns are stored as 1d numpy arrays.
    If you set row names, they're converted into a dictionary
    for fast access.

    There is a rich subselection/slicing API,
    see help(DataFrame.__get_item) (it also works for setting values).
    Please note that any slice get's you another DataFrame,
    to access individual entries use get_row(), get_column(), get_value().
    DataFrames also understand basic arithmetic and you can either
    add (multiply,...) a constant value, or another DataFrame of the same
    size / with the same column names.

    """
    def __init__(self, value_dict=None, columns_ordered=None, row_names_ordered = None, accept_as_is = False ):
        """Initialize with a dict of sequences as columns.

        Columns_ordered allows you to enforce an order of the columns,
        and may be partial (is passed to impose_partial_column_order()).

        """
        if value_dict is None:
            value_dict = {}
        if type(value_dict) is list:
            value_dict = swap_dictionary_list_axis(value_dict)
        try:
            self.num_rows = max(len(x) for x in value_dict.values())
        except TypeError as e:
            error_columns = {}
            for x in value_dict:
                try:
                    len(value_dict[x])
                except TypeError:
                    error_columns[x] = type(value_dict[x])
            raise TypeError("Columns %s did not contain types that can be converted to numpy 1d arrays. was: %s. Original error: %s" % (error_columns.keys(), error_columns, e))
        if (min(len(x) for x in value_dict.values()) != self.num_rows):
            lens = "\n".join(["%s: %s" % (len(value), key) for (key, value) in value_dict.items()])
            raise ValueError("Not all columns have the same length!: %s" % lens)

        if accept_as_is:
            self.value_dict = value_dict
        else:
            self.value_dict = {}
            for column in value_dict:
                try:
                    column = str(column)
                except:
                    column = bytes(column)
                try:
                    self.value_dict[(column)] = _convert_to_numpy(value_dict[column],
                                                                  column)
                except Exception as e:
                    raise ValueError("Could not convert to numpy %s - %s \n %s"%( column, e, value_dict[column]))

        self.columns_ordered = list(self.value_dict.keys())
        if not columns_ordered is None:
            self.impose_partial_column_order(columns_ordered)

        self._row_names_dict = None
        self._row_names_ordered = None
        self.row_names = row_names_ordered


    def rbind_copy(self, *others):
        """Stack frames below each other (rows)

        Take frames with the same fields, and stack them 'below' each other.
        Gives you a copy of the data.

        """
        new_data = self.value_dict.copy()
        for column_name in self.columns_ordered:
            new_data[column_name] = list(new_data[column_name])
        for other in others:
            for column_name in other.columns_ordered:
                if column_name not in self.columns_ordered:
                    raise NotImplementedError(
                        "No fix yet for when not all fields are in both frames: were \n%s, and \n%s" % (sorted(self.columns_ordered), sorted(other.columns_ordered)))
            for column_name in self.columns_ordered:
                new_data[column_name] = \
                    new_data[column_name] + list(other.value_dict[column_name])
        res = DataFrame(new_data, self.columns_ordered)
        return res

    def cbind_view(self, *others):
        """Stack frames next to each other ( column wise ).

        Take frames with distinct fields,
        but identical row lengths,
        and stack them next to each other in order.

        The new DataFrame shares the values with its parents.

        """
        #input checking
        already_seen = {}
        for column_name in self.columns_ordered:
            already_seen[column_name] = True
        for other in others:
            for column_name in other.value_dict.keys():
                if column_name in already_seen:
                    raise ValueError("Trying to cbind two DataFrames that both"
                                     " have the field '%s', you'll need to rename"
                                     " one." % column_name)
                already_seen[column_name] = True
        #assembling
        new_data = {}
        for column_name in self.columns_ordered:
            new_data[column_name] = self.value_dict[column_name]
        columns_ordered = self.columns_ordered[:]
        for other in others:
            for column_name in other.value_dict.keys():
                new_data[column_name] = other.value_dict[column_name]
            columns_ordered += other.columns_ordered
        return DataFrame(new_data, columns_ordered, accept_as_is = True)

    def join_columns_on(self, other, name_here, name_there):
        """Join to DataFrames on an column with common values"""
        if len(other) != len(self):
            raise ValueError("Dataframes did not contain the same number of rows: %i vs %i" % (len(self), len(other)))
        key_to_row_no_here = {}
        key_to_row_no_there = {}
        for ii, key in enumerate(self.get_column_view(name_here)):
            if key in key_to_row_no_here:
                raise KeyError("%s is not unique in this data frame, can't join on that" % name_here)
            key_to_row_no_here[key] = ii
        for ii, key in enumerate(other.get_column_view(name_there)):
            if key in key_to_row_no_there:
                raise KeyError("%s is not unique in other data frame, can't join on that" % name_there)
            key_to_row_no_there[key] = ii
        if set(key_to_row_no_here.keys()) != set(key_to_row_no_there):
            raise ValueError("Dataframes did not contain the same keys - can't join on that")
        new_columns = [x for x in other.columns_ordered if x != name_there]
        for x in new_columns:
            if x in self.columns_ordered:
                raise ValueError("Column %s already in this df!" % x)
        order = []
        for key in self.get_column_view(name_here):
            order.append(key_to_row_no_there[key])
        sorted_df = other[order,(new_columns)]
        return self.cbind_view(sorted_df)

    def insert_column(self, column_name, values, position='last'):
        """Insert a new column into your DataFrame.

        """
        if not len(values) == self.num_rows:
            raise ValueError(
                "New column did not have the same length as the"
                " DataFrame: %i vs %i" % (len(values), self.num_rows))
        if column_name in self.columns_ordered:
            raise ValueError("Column already exists: %s" % column_name)
        if position == 'last':
            self.columns_ordered.append(column_name)
        else:
            self.columns_ordered.insert(position, column_name)
        self.value_dict[column_name] = _convert_to_numpy(values, column_name)
        return self

    def drop_column(self, column_name):
        """Remove a column from the DataFrame.

        """
        self.columns_ordered.remove(column_name)
        del self.value_dict[column_name]
        return self

    def drop_all_columns_except(self, *column_names):
        """Remove all columns except those passed as par.ameters

        """
        save_names = list(column_names)
        for column_name in self.columns_ordered[:]:
            if column_name not in save_names:
                self.drop_column( column_name )
        return self

    def rename_column(self, old_name, new_name):
        """Rename a column

        old_name may also be the number of a column.

        """
        if old_name == new_name:
            return
        if new_name in self.value_dict:
            raise ValueError("Trying to rename %s into already existing column name %s. Drop %s first" % (old_name, new_name, new_name))
        old_name = self._map_column_number_to_name(old_name)
        try:
            self.value_dict[new_name] = self.value_dict[old_name]
        except KeyError:
            raise KeyError("Column %s not found. Available were %s" % (old_name, self.columns_ordered))
        del self.value_dict[old_name]
        idx = self.columns_ordered.index(old_name)
        self.columns_ordered[idx] = new_name
        return self

    def __str__(self):
        """Return a human readable representation.

        """
        def justify(str_value, width=20, just='center'):
            """Justify a string."""
            if len(str_value) > width:
                str_value = str_value[:width]
            if just == 'center':
                return str_value.center(width)
            elif just == 'left':
                return str_value.ljust(width)
            elif just == 'right':
                return str_value.rjust(width)
        buf = io.BytesIO()
        buf.write(justify(b'', width=5, just='right'))
        buf.write(b' ')
        for field in self.columns_ordered:
            buf.write(justify(field.encode('utf-8', "replace")))
            buf.write(b' ')
        buf.write(b'\n')

        for row in range(self.num_rows):
            if not self.row_names is None:
                buf.write(
                    justify(str(self.row_names[row]).encode('utf-8','replace'), width=10, just='left'))
                buf.write(b" ")
                buf.write(b" ")
            else:
                buf.write(justify(str(row).encode('utf-8','replace'), width=5, just='right'))
            for field in self.columns_ordered:
                buf.write(b" ")
                value = self.value_dict[field][row]
                if value is not None:
                    if isinstance(value, str):
                        v_str = '"%s"' % value
                    else:
                        try:
                            v_str = str(value)
                        except UnicodeEncodeError:
                            v_str = value.encode('utf-8', 'replace')
                else:
                    v_str = '--'
                buf.write(justify( v_str ).encode('utf-8', 'replace'))
                buf.write(b" ")
            buf.write(b"\n")
        buf.seek(0)
        return buf.read().decode('utf-8')

    def __repr__(self):
        """Return a python code representation."""
        res = ['DataFrame({']
        for k in self.value_dict:
            res.append(k + ': ')
            res.append(repr(self.value_dict[k]))
            res.append(',\n')
        res.append('},' + repr(self.columns_ordered) + ', ' + repr(self.row_names) + ')')
        return "".join(res)

    def get_row(self, row_idx):
        """Return a row as a dictionary."""
        res = {}
        row_idx = self._map_row_name_to_number(row_idx)
        for column_name in self.value_dict.keys():
            if (not hasattr(self.value_dict[column_name],'mask') or (not isinstance(self.value_dict[column_name].mask, bool) or not self.value_dict[column_name].mask[row_idx])):
                res[column_name] = self.value_dict[column_name][row_idx]
            else:
                res[column_name] = None
        return res

    def has_row(self, row_name):
        """check whether a certain row name or number exists"""
        try:
            row_idx = self._map_row_name_to_number(row_name)
            return row_idx < self.num_rows
        except KeyError:
            return False

    def get_row_as_list(self, row_idx):
        """Return a row as a list in order, without row names."""
        row_idx = self._map_row_name_to_number(row_idx)
        res = []
        for column_name in self.columns_ordered:
            if isinstance(self.value_dict[column_name], Factor):
                res.append(self.value_dict[column_name].map_value(self.value_dict[column_name][row_idx]))
            else:
                res.append(self.value_dict[column_name][row_idx])
        return res

    def get_column(self, column_name):
        """Return a column as a copy of the actual values.

        ->numpy array

        """
        column_name = self._map_column_number_to_name(column_name)
        return self.value_dict[column_name][:] #as to stay in line with the copy semantics

    def set_column(self, column_name, value):
        """Replace a column, or create a new one.

        Todo: Unittests"""
        column_name = self._map_column_number_to_name(column_name)
        value = _convert_to_numpy(value, column_name)
        if len(value) != self.num_rows:
            raise ValueError("New data for %s did not have the required number of columns (%i, was %i)" %
                             (column_name, self.num_rows, len(value)))
        if not column_name in self.value_dict:
            self.columns_ordered.append(column_name)
        self.value_dict[column_name] = value

    def get_column_view(self, column_name):
        """Returns a column directly from the internal storage.

        ->numpy array

        """
        column_name = self._map_column_number_to_name(column_name)
        try:
            return self.value_dict[column_name]
        except KeyError:
            import pprint
            raise KeyError("Column not found: %s, available: %s" % (column_name, pprint.pformat(self.columns_ordered)))

    def gcv(self, column_name):
        return self.get_column_view(column_name)

    def get_value(self, row_idx, column_name):
        """Return the value of a cell.

        ->int/str/object...
        To set, use df[row_idx, column_name] = X

        """
        column_name = self._map_column_number_to_name(column_name)
        row_idx = self._map_row_name_to_number(row_idx)
        if self.value_dict[column_name].mask[row_idx]:
            return None
        else:
            return self.value_dict[column_name][row_idx]

    def get_column_unique(self, column_name):
        return _find_levels(self.value_dict[column_name])

    def get_as_list_of_arrays_view(self):
        """Return the data storage as [numpy.ma.array, numpy.ma.array,...].

        """
        res = []
        for column_name in self.columns_ordered:
            res.append(self.value_dict[column_name])
        return res

    def get_as_list_of_lists(self):
        """Return the data storage as [ [..., ],... ].

        This is useful if you want to set from a DataFrame ignoring
        the column names::

            df_one = DataFrame({"A": (1, 2),'B': (3, 4)})
            df_two = DataFrame({"b": (1, 2), 'D': (3, 4)})
            df_one[:,:] =  df_two -> ValueError
            df_one[:, :] =  df_two.get_as_list_of_lists() => DataFrame({"A": (2, 4), "B": (6, 8) })

        """
        res = []
        for row_no in range(0, self.num_rows):
            res.append(self.get_row_as_list(row_no))
        return res

    def _map_column_number_to_name(self, col_number):
        """Map column number to name.

        No-op if it's not an int (i.e. already a name)
        """
        if type(col_number) == int:
            return self.columns_ordered[col_number]
        else:
            return col_number

    def _map_row_name_to_number(self, row_name):
        """Map a row name to a number.

        """
        if self._row_names_dict:
            try:
                return self._row_names_dict[row_name]
            except KeyError:
                pass
        try:
            row_idx = int(row_name)
            if row_idx > len(self):
                raise KeyError("")
            return row_idx
        except:
            if self._row_names_dict:
                raise KeyError("Could not find rowname: %s" % row_name)
            else:
                raise ValueError("This DataFrame has no row names, and that (%s) was no int convertible row number." % (row_name,))

    def _interpret_selection(self, selection):
        """ Turn whatever is passed in into two lists of rows and columns
            to select in that order.

            This is a bit of a mess, but I don't see any more elegant way.

        """
        columns_to_select = None
        rows_to_select = None

        if type(selection) == int: # a single number, so we return that row
            rows_to_select = numpy.zeros((len(self),), dtype=numpy.bool)
            rows_to_select[selection] = True
            columns_to_select = self.columns_ordered
        elif type(selection) == slice: # a request for rows, with no specification on the columns
            columns_to_select = self.columns_ordered
            if type(selection.start) == int: # we have an actual range. : is passed here as from 0 to magic constant...
                ind = selection.indices(self.num_rows)
                rows_to_select = range(ind[0], ind[1], ind[2])
            else: # neither a number, nor as slice request. this needs to be a sequence, either of int, then we can use it directly. or of boolean, which means we have to convert it into a list of row no.
            #but actually, we shouldn't get here if the user didn't do a[(1, 2, 3):] or a a[:]
                if selection.start is None and selection.stop is None and selection.step is None: #everything
                    columns_to_select = self.columns_ordered
                    rows_to_select = range(0, self.num_rows)
                else:
                    raise ValueError(
                        "Invalid argument passed into dataframe[] see"
                        " help(dataframe.__get__item)"
                        " (guess: You did not specify both dimensions)")

        elif len(selection) == 2: #a tuple with info for each dimension
            if type(selection[0]) == slice:
                if not ( selection[0].start is None and
                        selection[0].stop is None and
                        selection[0].step is None):
                    ind = selection[0].indices(self.num_rows)
                    rows_to_select = range(ind[0], ind[1], ind[2])
                else:
                    rows_to_select = range(0, self.num_rows)
            elif type(selection[0]) == int: # a single row
                rows_to_select = numpy.zeros((len(self),), dtype=numpy.bool)
                rows_to_select[selection[0]] = True
            elif isinstance(selection[0], six.string_types) or isinstance(selection[0], numpy.str_):
                rows_to_select = numpy.zeros((len(self),), dtype=numpy.bool)
                rows_to_select[self._map_row_name_to_number(selection[0])]  = True
            else:# neither a number/str, nor as slice request. this needs to be a sequence, either of int, then we can use it directly. or of boolean, which means we have to convert it into a list of row no.
                #if selection[0] is False:
                    #rows_to_select = numpy.zeros((self.num_rows,),dtype=numpy.bool)
                #elif selection[0] is True:
                    #rows_to_select = numpy.ones((self.num_rows,),dtype=numpy.bool)
                if len(selection[0]) == 0:
                    rows_to_select = numpy.zeros((self.num_rows,),dtype=numpy.bool)
                else:
                    if isinstance(selection[0][0],bool):
                        rows_to_select = numpy.array(
                            [i for i, p in enumerate(selection[0]) if p])
                    elif isinstance(selection[0][0],numpy.bool_):
                        rows_to_select = selection[0]
                    else: #that's a sequence of the rows we should include
                        rows_to_select = [self._map_row_name_to_number(x) for x in selection[0]]
            if type(selection[1]) == slice:
                if not ( selection[1].start is None and
                        selection[1].stop is None and selection[1].step is None):
                    ind = selection[1].indices(len(self.columns_ordered))
                    columns_to_select = range(ind[0], ind[1], ind[2])
                else:
                    columns_to_select = range(0, len(self.columns_ordered))
            elif type(selection[1]) == str or type(selection[1]) == int or type(selection[1]) == unicode:
                columns_to_select = (selection[1], )
            else: #some kind of sequence
                if type(selection[1][0]) in (bool, numpy.bool_):
                    columns_to_select = \
                            [i for i, p in enumerate(selection[1]) if p]
                else: #hopefully just an ordinary list of columns to actually use
                    columns_to_select = selection[1]
            columns_to_select = \
                    [self._map_column_number_to_name(x) for x in columns_to_select]
        else:
            raise ValueError("Invalid argument passend into"
                             " dataframe[], please make sure to specify both"
                             " dimensions (fill in ':' for anything you don't"
                             " want to choose in")
        return columns_to_select, rows_to_select

    def __getitem__(self, selection):
        """DateFrames handle an R inspired subset selection.

        Examples::

            df[0] - 1xX DateFrame
            df[0:2] - 2xX DateFrame
            df[0:8:2] - 2xX DateFrame, every second row.
            df[0, 'Field1'] = 1x1 DataFrame
            df[:, "Field2"] = Xby1 DataFrame
            df[:, ("Field2", "Field1")] = DataFrame containing just
                                          Field2, Field1 in the given order
            df[:, 0:1] - Xx2 DataFrame containing columns no. 0 and 1
            df[(0, 3, 5), (0, 2)] = 3x2 DataFrame with rows 0, 3 and 5 and
                                    the fields in order 0 and 2
            df[(True, True, False)] = 2xX DataFrame with just rows 0 and 1
            df[:, [x.startswith('shu') for x in df.columns_ordered]) =
                XxY DataFrame containing just the columns which's fieldnames
                start with 'shu'

        """
        columns_to_select, rows_to_select = self._interpret_selection(selection)
        #print columns_to_select, rows_to_select
        new_data = {}
        for column_name in columns_to_select:
            new_data[column_name] = self.value_dict[column_name][rows_to_select]
        #print new_data
        if self.row_names is None:
            row_names = None
        else:
            row_names = self.row_names[rows_to_select]
            if not isinstance(row_names, numpy.ndarray):
                row_names = [row_names]
        return DataFrame(new_data, columns_to_select, row_names_ordered = row_names)

    def __setitem__(self, selection, value):
        """Set values in a DataFrame.

        You can set single rows::

            df[3, "Field2"] = 55

        or columns with a sequence::

            df[:, "Field2"] = [1, 2, 3]

        or arbitary subsections with other DataFrames::

            df[(1, 2), ("Field1", Field2")] =
                DataFrame({"Field2": ..., "Field1": ...})

        (Note: Need to have identical column names, except if they're both single columned!)
        or from a nested sequence::

            df[(1, 2), ("Field1", Field2")] =
                [(3, 4), (5, 6)]

        (also see :py:func:`get_as_list_of_lists`)


        """
        columns_to_select, rows_to_select = self._interpret_selection(selection)
        if (
                len(columns_to_select) == 1 and
                len(rows_to_select) == 1 and
                not (isinstance(rows_to_select, numpy.ndarray) and rows_to_select.dtype == numpy.bool) and
                not (type(selection) == tuple and type(selection[0]) == slice and len(self) == 1)
        ):
            if (
                    isinstance(value, DataFrame) and
                    value.num_rows == 1 and
                    len(value.columns_ordered) == 1
            ):
                self.value_dict[columns_to_select[0]][rows_to_select[0]] = \
                        value.value_dict[value.columns_ordered[0]][0]
            else:
                self.value_dict[columns_to_select[0]][rows_to_select[0]] = value
        else:
            if len(columns_to_select) == 1: #but we have selected multiple rows...
                if isinstance(value, DataFrame):
                    if isinstance(rows_to_select, numpy.ndarray) and rows_to_select.dtype == numpy.bool:
                        if value.num_rows != numpy.sum(rows_to_select):
                            raise ValueError(
                                "Unequal number of rows in dataframe"
                                " and subset of dataframe being set (bool).")
                    else: #not a boolean selection array
                        if value.num_rows != len(rows_to_select):
                            raise ValueError(
                            "Unequal number of rows in dataframe"
                            " and subset of dataframe being set.")

                    if len(value.columns_ordered) != 1:
                        raise ValueError("Trying to set a single"
                                         " column from a DataFrame"
                                         " with more than one column")
                    self.value_dict[columns_to_select[0]][rows_to_select] = \
                            value.value_dict[value.columns_ordered[0]]
                else:
                    self.value_dict[columns_to_select[0]][rows_to_select] = \
                            value
            else:
                if isinstance(value, DataFrame):
                    for column_name in value.columns_ordered:
                        if not column_name in columns_to_select:
                            raise ValueError(
                                "DataFrame._setitem_ contained a value (%s)"
                                " that is not in the dataframe being set."
                                % column_name)
                    for column_name in columns_to_select:
                        if not column_name in value.columns_ordered:
                            raise ValueError("Setting a row in a dataframe "
                                             "where a field was missing"
                                             " (%s)" % column_name)
                    if isinstance(rows_to_select, numpy.ndarray) and rows_to_select.dtype == numpy.bool:
                        rows_to_select = numpy.where(rows_to_select)[0]
                    if len(rows_to_select) != value.num_rows:
                        raise ValueError("Unequal number of rows in dataframe "
                                         "and subset of dataframe being set.")
                    for column_name in columns_to_select:
                        for other_row, self_row in enumerate(rows_to_select):
                            self.value_dict[column_name][self_row] = \
                                    value.value_dict[column_name][other_row]
                elif isinstance(value, dict):
                    for column_name in value:
                        if not column_name in columns_to_select:
                            raise ValueError(
                                "DataFrame._setitem_ contained a column (%s)"
                                " that is not in the dataframe being set."
                                % column_name)
                    for column_name in columns_to_select:
                        if not column_name in value:
                            raise ValueError(
                                "Setting a row in a dataframe where a column"
                                " was missing (%s)" % column_name)
                    if isinstance(rows_to_select, numpy.ndarray) and rows_to_select.dtype == numpy.bool:
                        rows_to_select = numpy.where(rows_to_select)[0]
                    if len(rows_to_select) == 1: #we're setting a single row from a key value dict
                        for column_name in columns_to_select: #we're setting a single row from a key value dict
                            try:
                                self.value_dict[column_name][rows_to_select[0]] = \
                                        value[column_name]
                            except ValueError as e:
                                raise ValueError("Error setting column %s" % column_name, e)
                    else:
                        if len(rows_to_select) != len(next(iter(value.values()))):
                            raise ValueError(
                                "Unequal number of rows when setting subset"
                                " of DataFrame")
                        for column_name in columns_to_select:
                            for value_row, self_row in enumerate(rows_to_select): #we're setting multiple rows from a key -> sequence dict (similar to construction)
                                self.value_dict[column_name][self_row] = \
                                        value[column_name][value_row]
                else: #let's treat it as a (hopefully appropriate) sequence)
                    if isinstance(rows_to_select, numpy.ndarray) and rows_to_select.dtype == numpy.bool:
                        rows_to_select= numpy.where(rows_to_select)[0]
                    if len(rows_to_select) == 1:
                        if len(value) != len(self.columns_ordered):
                            raise ValueError(
                                "Setting a row from a sequence with the"
                                " wrong number of values")
                        for o_row, s_col in enumerate(columns_to_select):
                            self.value_dict[s_col][rows_to_select[0]] = \
                                    value[o_row]
                    else:
                        for row in value:
                            if len(row) != len(self.columns_ordered):
                                raise ValueError(
                                    "Setting a row from a sequence with"
                                    " the wrong number of values")
                        print(rows_to_select)
                        for o_col, s_col in enumerate(columns_to_select):
                            for o_row, s_row in enumerate(rows_to_select):
                                self.value_dict[s_col][s_row] = \
                                        value[o_row][o_col]

    def __iter__(self):
        """Iterate a single column DataFrame.

        Raises ValueError on more than one column.
        Please use these instead::

            df.iter_values_columns_first()
            df.iter_values_rows_first(),
            df.iter_rows()
            df.iter_columns()
        """
        try:
            self._assert_single_columned()
            return self.iter_values_columns_first()
        except ValueError:
            raise ValueError("Please use one of the iter_xyz functions. See help(DatFrame.__iter)")

    def iter_values_columns_first(self):
        """Return an iterator over all values.

            Iterates first column one, first row,
                     then column one, second row...
        """
        for column_name in self.columns_ordered:
            for value in self.value_dict[column_name]:
                yield value

    def iter_values_rows_first(self):
        """Return an iterator over all values.

            Iterates first row column one,
                     first row column two...
        """
        for i in range(0, self.num_rows):
            for column_name in self.columns_ordered:
                yield self.value_dict[column_name][i]

    def iter_rows(self):
        """Return an iterator over the rows (dicts).

        """
        def inner():
            for i in range(0, self.num_rows):
                yield self.get_row(i)
        return IteratorPlusLength(inner(), self.num_rows)

    def iter_rows_old(self):
        """Return an iterator over the rows (dicts).

        """
        for i in range(0, self.num_rows):
            yield self.get_row(i)

    def iter_rows_as_list(self):
        for i in range(0, self.num_rows):
            yield self.get_row_as_list(i)


    def iter_columns(self):
        """Return an iterator over the columns (arrays).

        """
        for column_name in self.columns_ordered:
            yield self.value_dict[column_name]

    def groupby(self, column_name):
        """Yield (value, sub_df) for all values of column column_name, where sub_df[:,'column_name'] == value for each subset.
        Basically, itertools.groupby for dataframes.
        No prior sorting necessary.

        """
        sorted_df = self.sort_by(column_name)
        column = sorted_df.get_column_view(column_name)
        pos, = numpy.where(column[1:] != column[:-1])
        pos = numpy.concatenate(([0],pos+1,[len(column)]))
        for start, stop in izip(pos[:-1],pos[1:]):
            value = column[start]
            sub_df = sorted_df[start:stop, :]
            yield (value, sub_df)




    def __len__(self):
        """Return the row count of the DataFrame."""
        return self.num_rows

    def dim(self):
        """Return (rowCount, columnCount)

        For R compability.
        """
        return (self.num_rows, len(self.columns_ordered))

    def _assert_single_columned(self):
        """Abstraction of the error that occurs if
        you try to compare a dataframe with more than one column"""
        if len(self.columns_ordered) > 1:
            raise ValueError("The truthvalue of comparing more"
                             " than one column is not definied. Please use DataFrame.where(lambda row:...)")

    def where(self, boolean_row_function):
        """Return numpy.array(dtype=bool,
            data=[boolean_row_function(x) for x in self.iter_rows())

            I.e. return a truth array, iterate over all rows
            and call boolean_row_function on each to fill array.

        """
        result = numpy.zeros((self.num_rows, ), dtype=numpy.bool)
        for row_number in range(0, self.num_rows):
            row = {}
            for column_name in self.value_dict:
                row[column_name] = self.value_dict[column_name][row_number]
            if boolean_row_function(row):
                result[row_number] = True
        return result

    def __gt__(self, other):
        self._assert_single_columned()
        return self.where(lambda row: row[self.columns_ordered[0]] > other)

    def __ge__(self, other):
        self._assert_single_columned()
        return self.where(lambda row: row[self.columns_ordered[0]] >= other)

    def __lt__(self, other):
        self._assert_single_columned()
        return self.where(lambda row: row[self.columns_ordered[0]] < other)

    def __le__(self, other):
        self._assert_single_columned()
        return self.where(lambda row: row[self.columns_ordered[0]] <= other)

    def __eq__(self, other):
        """Equality for DataFrames with other DataFrames and sequences.

        Sequence equality for single column DataFrames only.

        """
        if isinstance(other, DataFrame):
            return self.__eq__dataframes(other)
        else:
            return self.__eq__values(other)

    def __eq__values(self, other):
        """Compare the single column of this frame with the sequence other"""
        self._assert_single_columned()
        if self.value_dict[self.columns_ordered[0]].dtype.type ==  numpy.string_:
            other = str(other)

        try:
            #return self.value_dict[self.columns_ordered[0]] == other
            return numpy.equal(self.value_dict[self.columns_ordered[0]], other)
        except AttributeError as e:
            if str(e) == "'bool' object has no attribute 'view'" and other is None:
                #we have run into a bug in the masked array... let's do this by hand
                result = numpy.zeros((self.num_rows,), dtype=numpy.bool)
                mask = self.value_dict[self.columns_ordered[0]].mask
                for ii, value in enumerate(self.value_dict[self.columns_ordered[0]]):
                    if value is None or mask[ii]:
                        result[ii] = True
                return result
            else:
                raise

    def __ne__(self, other):
        if isinstance(other, DataFrame):
            return not self.__eq__dataframes(other)
        else:
            return ~ self.__eq__values(other)

    def is_nan(self):
        self._assert_single_columned()
        result = numpy.zeros((self.num_rows, ), dtype=numpy.bool)
        col = self.get_column_view(0)
        for row_number in range(0, self.num_rows):
            result[row_number] = numpy.isnan(col[row_number])
        return result

    def __eq__dataframes(self, other):
        """Two dataframes are equal if they contain the same columns
           (in any order) and have the same data in the columns, ordered the same way,
           and have the same row names)
        """
        if type(other) == type(self):
            colnames = self.columns_ordered[:]
            colnames.sort()
            other_colnames = other.columns_ordered[:]
            other_colnames.sort()
            if other_colnames != colnames:
                return False
            if other.num_rows != self.num_rows:
                return False
            if not (self.row_names is None and other.row_names is None):
                if other.row_names is None and not self.row_names is None:
                    return False
                if self.row_names is None and not other.row_names is None:
                    return False
                if (other.row_names != self.row_names).any():
                    return False
            for k in self.value_dict:
                mine = self.value_dict[k]
                others = other.value_dict[k]
                for i in range(0, self.num_rows):
                    try:
                        if mine[i] != others[i] and not (numpy.isreal(mine[i]) and numpy.isreal(others[i]) and numpy.isnan(mine[i]) and numpy.isnan(others[i])):
                            return False
                    except AttributeError:
                        try:
                            if not(mine[i].mask and others[i].mask):
                                return False
                        except:
                            return False
            return True

    def __arithmetic__(self, other, operator, human_readable_operator_name):
        """Refactor calculations on (DataFrame op DataFrame)
            and (DataFrame op value).

        """
        if isinstance(other, DataFrame):
            if other.num_rows != self.num_rows:
                raise ValueError("Arithmetic with unequal number of rows.")
            for column_name in self.columns_ordered:
                if not column_name in other.columns_ordered: #ok, field names don't match, but we can still do this by position
                    if len(self.columns_ordered) != len(other.columns_ordered):
                        raise ValueError("Arithmetic with dataframe"
                                         " with unequal field names"
                                         " and unequal field number")
                    new_data_dict = {}
                    names = []
                    for column_number in range(0, len(self.columns_ordered)):
                        name = ('(' + self.columns_ordered[column_number] +
                                human_readable_operator_name + other.columns_ordered[column_number] + ")")
                        names.append(name)
                        new_data_dict[name] = \
                                operator(self.value_dict[self.columns_ordered[column_number]],
                                         other.value_dict[other.columns_ordered[column_number]])
                    return DataFrame(new_data_dict, names, accept_as_is = True)
            for k in other.columns_ordered:
                if not k in self.columns_ordered:
                    raise ValueError(
                        "Arithmetic with dataframe with unequal fields")
            new_data_dict = {}
            for k in self.columns_ordered:
                new_data_dict[k] = operator(self.value_dict[k],
                                            other.value_dict[k])
            return DataFrame(new_data_dict, self.columns_ordered, accept_as_is = True)
        else:
            new_data = {}
            for column_name in self.columns_ordered:
                new_data[column_name] = operator(self.value_dict[column_name], other)
            return DataFrame(new_data, self.columns_ordered, accept_as_is = True)

    def __mul__(self, other):
        return self.__arithmetic__(other, numpy.ndarray.__mul__, '*')

    def __add__(self, other):
        return self.__arithmetic__(other, numpy.ndarray.__add__, '+')

    def __sub__(self, other):
        return self.__arithmetic__(other, numpy.ndarray.__sub__, '-')

    def __div__(self, other):
        return self.__arithmetic__(other, numpy.ndarray.__div__, '/')

    def __floordiv__(self, other):
        return self.__arithmetic__(other, numpy.ndarray.__floordiv__, '//')

    def __mod__(self, other):
        return self.__arithmetic__(other, numpy.ndarray.__mod__, '%')

    def __truediv__(self, other):
        return self.__arithmetic__(other, numpy.ndarray.__truediv__, '/')

    def __abs__(self):
        res = self.copy()
        for k in res.value_dict:
            res.value_dict[k] = numpy.abs(res.value_dict[k])
        return res

    def sort_by(self, field_or_fields, ascending=True):
        """Sort a DataFrame by one or more fields.

        Direction (ascending) needs to be specified for each field.
        This is not an inplace sort, but returns a new DataFrame!

        You can sort by one field: sort_by(my_field, ascending=False)
        or by multiple fields: sort_by([my_fieldA, my_fieldB], [True, False]),
        but then you will need to pass in the order for each one as well.


        """
        #input validation and converting into sequences
        if self.num_rows == 0:
            return self.copy()
        if type(field_or_fields) is str or type(field_or_fields) is int:
            field_or_fields = (field_or_fields, )
        elif (not (type(field_or_fields) is tuple or
                   type(field_or_fields) is list)):
            raise ValueError(
                "field_or_fields must be eiher an int,"
                " a str, or a list or tuple of either")
        if type(ascending) is bool:
            ascending = (ascending, )
        if (len(ascending) != len(field_or_fields)):
            raise ValueError("You have to specify as many"
                             " directions in ascending (list)"
                             " as you have fields you're sorting on")
        field_or_fields = \
                [self._map_column_number_to_name(x) for x in field_or_fields]
        for f in field_or_fields:
            if not f in self.columns_ordered:
                raise KeyError("Unknown column_name: %s" % f)
        ziped = list(zip(field_or_fields, ascending))
        neworder = list(range(0, self.num_rows))
        #compare to values at a given index
        def compare_by_index(indexa, indexb):
            """Compare to rows by the fields in the specified order."""
            for field, asc in ziped:
                try:
                    a = self.value_dict[field][indexa]
                    b = self.value_dict[field][indexb]
                    compresult = int(a>b)-int(a<b)
                except AttributeError as e:
                    if str(e).find("'NotImplementedType' object has no attribute 'view'") != -1:
                        compresult = -1
                    else:
                        raise e
                if compresult != 0:
                    if asc:
                        return compresult
                    else:
                        return compresult * -1
            return 0
        #and now, sort the newOrder and create a new dataframe just like it
        neworder.sort(key=functools.cmp_to_key(compare_by_index))
        res = self[neworder, :]
        return res

    def mean(self, field):
        """Return the arithmetic mean (average) of a column."""
        values = self.get_column_view(field)
        return numpy.mean(values)

    def mean_and_std(self, field):
        """Return mean and standard deviation."""
        values = self.get_column_view(field)
        return numpy.mean(values), numpy.std(values)

    def mean_and_sem(self, field):
        """Return mean and standard error of the mean."""
        values = self.value_dict[field]
        n = len(values)
        return numpy.mean(values), numpy.std(values)/float(numpy.sqrt(n))

    def get_column_names(self):
        """Return the column names (in order)."""
        return self.columns_ordered[:] # return copy

    def as_2d_matrix(self, dtype=None):
        """Return all columns as 2d(nRows, nCols)-numpy matrix.

        Default dtype is float64
        Raises a ValueError if not all columns could be converted.
        """
        return self.as2DMatrix()

    def as2DMatrix(self, dtype=None):
        """Return all columns as 2d(nRows, nCols)-numpy matrix.
        Please use pep 8 conform as_2d_matrix()

        Default dtype is float64
        Raises a ValueError if not all columns could be converted.
        """

        if dtype is None:
            dtype = numpy.float
        res = numpy.zeros(self.dim(),dtype=numpy.float64)
        for colno, columnname in enumerate(self.columns_ordered):
            res[:,colno] = self.value_dict[columnname]
        return res

    def turn_into_level(self, column_name, levels = None):
        """Convert a column into something that fit's into an R factor"""
        if isinstance(self.value_dict[column_name], Factor):
            return
        self.value_dict[column_name] = Factor(self.value_dict[column_name], levels)

    def turn_into_character(self, column_name, levels = None):
        """Convert a level column into it's character values"""
        if not isinstance(self.value_dict[column_name], Factor):
            return
        self.value_dict[column_name] = self.value_dict[column_name].as_levels()

    def digitize_column(self, column_name, bins = None, no_of_bins = None, min=None, max = None):
        """Convert a column into a number of bin-ids"""
        if bins is None:
            if no_of_bins is None:
                raise ValueError("digitize_column needs either a set of bins, or at least a number of equal sized bins to create")
            if min is None:
                min = numpy.min(self.value_dict[column_name])
            if max is None:
                max = numpy.max(self.value_dict[column_name])
            bins = numpy.array(range(min, max, (max - min) // no_of_bins))[1:] #so that anything < min + one step get's 0
        self.value_dict[column_name] = _convert_to_numpy(numpy.digitize(self.value_dict[column_name], bins), column_name)

    def rankify_column(self, column_name, lower_is_better = True):
        """Turn a column into a ranked order 0..len(self)"""
        to_sort = []
        for ii, value in enumerate(self.value_dict[column_name]):
            to_sort.append((value, ii))
        to_sort.sort()
        if not lower_is_better:
            to_sort.reverse()
        new_col = numpy.zeros((self.num_rows,), dtype=numpy.int32)
        for new_rank, (value, old_rank) in enumerate(to_sort):
            new_col[old_rank] = new_rank
        self.value_dict[column_name] = new_col

    def rescale_column_0_1(self, column_name):
        col = self.get_column_view(column_name)
        min_value = numpy.min(col)
        max_value = numpy.max(col)
        col[:] = (col - min_value) * 1.0 / (max_value - min_value)


    def copy(self):
        """Return a deep copy of the DataFrame.
        """
        vd = {}
        for key in self.value_dict:
            vd[key] = self.value_dict[key].copy()
        return DataFrame(vd, self.columns_ordered[:], self.row_names, accept_as_is=True)

    def shallow_copy(self):
        """Return a shallow copy of the DataFrame - ie. shared data columns, but differing objects"""
        return DataFrame(self.value_dict.copy(), self.columns_ordered[:], self.row_names, accept_as_is=True)


    def types(self):
        """Return a tuple of the types of the DataFrame"""
        return tuple(((key, self.value_dict[key].dtype) for key in self.value_dict))

    def convert_type(self, column_name, value_casting_func):
        """Cast a column into another type"""
        new_col = []
        for value in self.value_dict[column_name]:
            new_col.append(value_casting_func(value))
        new_col = _convert_to_numpy(new_col, column_name)
        self.value_dict[column_name] = new_col

    def __getstate__(self):
        """Pickle dataframe"""
        state = {}
        for k in self.__dict__:
            if k != 'value_dict':
                state[k] = self.__dict__[k]
            else:
                state['value_dict_pickle'] =  {}
                for key in self.value_dict:
                    op_data = io.BytesIO()
                    op_mask = io.BytesIO()

                    #string columns are pickled as a list of strings.
                    #and zipped.
                    #this enlarges the other pickles by one byte,
                    #but drastically reduces the size on disk for
                    #string columns that have few very large strings
                    #but a large collection of shorter strings

                    if ((self.value_dict[key].dtype.str.startswith('|O')
                        and (len(self) > 0 and isinstance(self.value_dict[key][0],str) ))
                            or self.value_dict[key].dtype.str.startswith("|S")):
                        op_data.write(b's')
                        op_data.write(zlib.compress( pickle.dumps(tuple([str(x) for x in self.value_dict[key]]), pickle.HIGHEST_PROTOCOL)))
                    elif self.value_dict[key].dtype.str.startswith('|O'): #not as string, but an object...
                            op_data.write(b'o')
                            numpy.save(op_data, numpy.ma.getdata(self.value_dict[key]))
                    else:
                        op_data.write(b'n')
                        numpy.save(op_data, numpy.ma.getdata(self.value_dict[key]))
                    mask = numpy.ma.getmask(self.value_dict[key])
                    if mask.all():
                        op_mask.write(b't')
                    elif (~mask).all():
                        op_mask.write(b'f')
                    else:
                        op_mask.write(b'a')
                        numpy.save(op_mask, mask)
                    state['value_dict_pickle'][key] = (op_data.getvalue(), op_mask.getvalue())
        return state

    def __setstate__(self, state):
        """Unpickle dataframe"""
        for k in state:
            if k != 'value_dict_pickle':
                setattr(self, k, state[k])
            else:
                self.value_dict = {}
                for key in state[k]:
                    op_data, op_mask = io.BytesIO(state[k][key][0]),io.BytesIO(state[k][key][1])
                    data_type = op_data.read(1)
                    if data_type == b's':
                        comp = op_data.read()
                        z = zlib.decompress(comp)
                        try:
                            tups = pickle.loads(z)
                            data = numpy.array(tups,dtype=numpy.object)
                        except:
                            import pdb
                            pdb.post_mortem()
                    elif data_type == b'o':
                        data = numpy.load(op_data)
                    elif data_type == b'n':
                        data = numpy.load(op_data)
                    else: #we're abusing that pickled numpy data starts with \x93NUMPY to preserve backwards compability
                        op_data.seek(0,0)
                        data = numpy.load(op_data)
                    mask_type = op_mask.read(1)
                    if mask_type == b't':
                        mask = True
                    elif mask_type == b'f':
                        mask = False
                    elif mask_type == b'a':
                        mask = numpy.load(op_mask)
                    else:
                        op_mask.seek(0,0)
                        mask = numpy.load(op_mask)
                    self.value_dict[key] = numpy.ma.core.MaskedArray(data,dtype=data.dtype)
                    self.value_dict[key].mask = mask
                pass

    def impose_partial_column_order(self, order, last_order = None):
        """Order columns... first those in order, then everything not in order or last_order alphabetically, then last_order"""
        if not last_order:
            last_order = []
        for o in (order, last_order):
            for column_name in o:
                if not column_name in self.columns_ordered:
                    raise KeyError("Column does not exist: %s" % (column_name,))
        remaining = []
        for column_name in self.columns_ordered:
            if not column_name in order and not column_name in last_order:
                remaining.append(column_name)
        remaining.sort()
        columns_ordered = order[:]
        columns_ordered.extend(remaining)
        columns_ordered.extend(last_order)
        self.columns_ordered = columns_ordered

    def melt(self, id_vars, measure_vars, measurement_name_target = 'Measurement', value_target = 'Value' ):
        if isinstance(id_vars, str):
            raise ValueError("id_vars needs to be alist, not a string")
        result = { measurement_name_target: [], value_target: []}
        for id_var in id_vars:
            result[id_var] = []
        if measure_vars is None:
            measure_vars = [x for x in self.columns_ordered if not x in id_vars]
        for row in self.iter_rows():
            for measure_var in measure_vars:
                for id_var in id_vars:
                    result[id_var].append(row[id_var])
                result[measurement_name_target].append(measure_var)
                result[value_target].append(row[measure_var])
        result = DataFrame(result)
        for id_var in id_vars:
            if isinstance(self.value_dict[id_var], Factor):
                result.value_dict[id_var] = Factor(result.value_dict[id_var], self.value_dict[id_var].levels, False)
        return result

    def aggregate(self, key_vars, aggregation_function):
        """Iterate for every value combination of the key vars, call the aggregation_function with the sub-df. Take the returned dicts, turn them into a new df"""
        if not hasattr(key_vars, '__iter__'):
            raise ValueError("key_vars must be a list/iterable")
        key_vars = tuple(key_vars)
        df = self.sort_by(key_vars, [True] *len(key_vars))
        last_key = None
        last_start = 0
        res = []
        for ii in range(0, len(df)):
            key = tuple(df[ii, key_vars].get_row_as_list(0))
            if key != last_key:
                if last_key != None:
                    sub_df = df[last_start:ii,:]
                    r = aggregation_function(
                        sub_df)
                    if not isinstance(r, dict):
                        raise ValueError("aggregation function did not return a dict for the new row")
                    for k in key_vars:
                        r[k] = df.get_value(ii -1, k)
                    res.append(r)
                    last_start = ii
                last_key = key
        if last_start != len(df):
            r = aggregation_function(
                        df[last_start:,:])
            for k in key_vars:
                r[k] = df.get_value(len(df) -1, k)
            res.append(r)
        return DataFrame(res)

    def sort_by_minimal_distance(self, columns_to_consider, distance_function = 'euclid'):
        if distance_function == 'euclid':
            def euclid(a, b):
                sum = 0
                for ii in range(0, len(a)):
                    sum += (a[ii] - b[ii])**2
                return sum
            distance_function = euclid
        final_order = []
        temp_order = range(0, len(self))
        final_order.append(temp_order.pop())
        lookat = self[:, columns_to_consider]
        while temp_order:
            shortest_distance = sys.maxint
            select = None
            col_current = lookat[final_order[-1], :].as2DMatrix()
            for name in temp_order:
                col_test = lookat[name, :].as2DMatrix()
                distance = distance_function(col_current, col_test)
                if distance < shortest_distance:
                    shortest_distance = distance
                    select = name
            final_order.append(select)
            temp_order.remove(select)
        return self[final_order, :]

    def to_pandas(self):
        import pandas
        return pandas.DataFrame(self.value_dict)[self.columns_ordered]

def from_pandas(pandas_df):
    return DataFrame(pandas_df.to_dict('list'))

class _ShapeAttribute(object):

    def __get__(self, obj, type=None):
        return obj.dim()

    def __set__(self, obj, type=None):
        raise ValueError("It is meaningless to set the shape of a DataFrame")

    def __delete__(self, obj):
        raise ValueError("It is meaningless to delete the shape of a DataFrame")
DataFrame.shape = _ShapeAttribute()

class _RowNamesAttribute(object):
    """Handle the row names ordered and as a
    dict at the same time.

    """
    def __get__(self, obj, type=None):
        return obj._row_names_ordered

    def __set__(self, obj, value):
        if not value is None:
            if len(value) != obj.num_rows:
                raise ValueError("Invalid number of rownames. Should be %i, was %i." % (obj.num_rows, len(value)))
            if isinstance(value, DataFrame):
                value._assert_single_columned()
                value = value.get_column_view(0)
            arr = numpy.array(value)
            if arr.dtype.kind == 'i' or arr.dtype.kind == 'u':
                raise ValueError("Integer rownames are not supported")
            obj._row_names_ordered = arr
            self.update_dictionary(obj)
        else:
            obj._row_names_ordered = None
            obj._row_names_dict = None

    def __del__(self, obj):
        obj._row_names_ordered = None

    def update_dictionary(self, obj):
        if not obj._row_names_ordered is None:
            lookup = {}
            try:
                for (row_no, value) in enumerate(obj._row_names_ordered):
                    if value in lookup:
                        raise ValueError("Duplicate row name: %s" % value)
                    else:
                        lookup[value] = row_no
            except: #happens if 'value in lookup' fails (not hashable?)
                print('issue is with %s' % value)
                raise
            obj._row_names_dict = lookup
DataFrame.row_names = _RowNamesAttribute()

def DataFrameFrom2dArray(array, column_names, row_names = None):
    if array.ndim != 2:
        raise ValueError("Array needs to be 2 dimensional")
    value_dict = {}
    if len(column_names) != array.shape[1]:
        raise ValueError("len(column_names) != 2nd dimension of the array")
    if not row_names is None and len(row_names) != array.shape[0]:
        raise ValueError("len(row_names) != array.shape[0]")
    for i, col_name in enumerate(column_names):
        value_dict[col_name] = array[:,i]
    return DataFrame(value_dict, column_names, row_names)

def combine(dataframe_generator):
    """rowbind multiple dataframes into one"""
    all_dfs = list(dataframe_generator)
    total_row_count = 0
    columns = set()
    dtypes = {}
    for df in all_dfs:
        if not columns:
            columns = set(df.columns_ordered)
        else:
            if columns != set(df.columns_ordered):
                raise KeyError("Dataframes had unequal columns")
        for column_name in df.columns_ordered:
            if not column_name in dtypes:
                dtypes[column_name] = df.get_column_view(column_name).dtype
            else:
                dtypes[column_name] = numpy.find_common_type([], [dtypes[column_name], df.get_column_view(column_name).dtype])
        total_row_count += len(df)
    value_dict = {}
    for column_name in columns:
        value_dict[column_name] = numpy.ma.zeros((total_row_count, ), dtype=dtypes[column_name])
    offset = 0
    for df in all_dfs:
        for column_name in columns:
            value_dict[column_name][offset: offset + len(df)] = df.value_dict[column_name]
        offset += len(df)
    return DataFrame(value_dict, all_dfs[0].columns_ordered)


if sys.version_info[0] == 2:
    unicode_string = unicode
    long_int = long
else: # python3 is the future
    unicode_string = str
    long_int = int


def _convert_to_numpy(seq, column_name):
    """Convert a sequence to an appropriate numpy array

    This is fairly dump:
        all ints -> numpy.int32
        all floats -> numpy float
        all complex -> numpy.cfloat
        string -> S%i % max(len(...))
        [] -> numpy.uint8
        all int, float -> numpy.float
        all something, str -> s%i % max(len(str(...)))
    """
    type_ = type(seq)
    #here's what we do if it's already a numpy array
    if type_ is numpy.ma.core.MaskedArray:
        return seq
    elif isinstance(seq, Factor):
        return seq
    elif isinstance(seq, numpy.ndarray):
        res =  numpy.ma.core.MaskedArray(seq)
        res.mask = False
        return res
    #and now, for the heuristic
    types_present = {}
    for value in seq:
        if not value is None:
            types_present[type(value)] = True

    for type_ in types_present.keys():
        if str(type_) == "<type 'rpy2.rinterface.NAIntegerType'>":  # we don't want to import rpy2 at this location
            seq = _replace_R_NAInteger_with_None(seq)
    #do in this order
    #any string -> all strings
    #any float64 -> all floats
    #any smaller float -> float
    #any large int -> large int (python ints = numpy.int32)
    #any large uint -> large uint
    #any bool -> bolo
    #other -> object
    zero_value = 0
    if (unicode_string in types_present):
        seq = [None if (s is None or (isinstance(s, numpy.ma.MaskedArray) and s.mask)) else
                        (s if isinstance(s, unicode_string) else unicode_string(s)) for s in seq]
        max_length = max(len(s) for s in seq if not s is None)
        dtype = numpy.object
        zero_value = ''
    elif (str in types_present or numpy.string_ in types_present):
        max_length = max(len(str(s)) for s in seq if not s is None)
        seq = [None if (s is None or (isinstance(s, numpy.ma.MaskedArray) and s.mask)) else
                        (s if isinstance(s, str) else str(s)) for s in seq]
        #seq = [s is None and "" or str(s) for s in seq]
        dtype = numpy.object
        zero_value = ''
    elif (numpy.float64 in types_present):
        dtype= numpy.float64
    elif float in types_present or numpy.float in types_present:
        dtype = numpy.float
    elif complex in types_present or numpy.cfloat in types_present:
        dtype = numpy.cfloat
    elif long_int in types_present or numpy.int64 in types_present:
        dtype = numpy.int64
    elif numpy.uint64 in types_present:
        dtype = numpy.uint64
    elif int in types_present:
        max_int_value = 2147483647
        dtype = numpy.int32
        for value in seq:
            if value > max_int_value:
                dtype = numpy.int64
                break
    elif numpy.int32 in types_present:
        dtype = numpy.int32
    elif numpy.uint32 in types_present:
        dtype = numpy.uint32
    elif numpy.int8 in types_present:
        dtype = numpy.int8
    elif numpy.uint8 in types_present:
        dtype = numpy.uint8
    elif bool in types_present or numpy.bool in types_present:
        dtype = numpy.bool
    else:
        dtype = numpy.object



    try:
        res = numpy.ma.zeros(len(seq), dtype=numpy.dtype(dtype))
    except TypeError as e:
        raise TypeError(e.message + ' column name %s, dtype: %s' % ( column_name, dtype))
    res[:] = [(not x is None) and x or zero_value for x in seq]
    res.mask = [(x is None or (hasattr(x,'mask') and x.mask)) for x in seq]
    return res

def swap_dictionary_list_axis(alist):
    result = {}
    for field in alist[0].keys():
        result[field] = []
    for row in alist:
        for field in row:
            result[field].append(row[field])
    return result


class IteratorPlusLength:

    def __init__(self, generator, length):
        self.length = length
        self.generator = generator

    def next(self): # python2
        return self.generator.next()

    def __next__(self): # python3
        return next(self.generator)


    def __len__(self):
        return self.length

    def __iter__(self):
        return self


try:
    import rpy2.robjects as ro
    import rpy2.robjects.conversion
    import rpy2.rinterface as rinterface
    import rpy2.robjects.numpy2ri
    previous_py2ri = ro.conversion.py2ri

    def numpy2ri_vector(o):
        """Convert a numpy 1d array to an R vector.

        Unlike the original conversion which converts into a list, apperantly."""
        if len(o.shape) != 1:
            raise ValueError("Dataframe.numpy2ri_vector can only convert 1d arrays")
        if isinstance(o, Factor):
            res = ro.r['factor'](o.as_levels(), levels=o.levels, ordered=True)
        elif isinstance(o, numpy.ndarray):
            if not o.dtype.isnative:
                raise(ValueError("Cannot pass numpy arrays with non-native byte orders at the moment."))

            # The possible kind codes are listed at
            #   http://numpy.scipy.org/array_interface.shtml
            kinds = {
                # "t" -> not really supported by numpy
                "b": rinterface.LGLSXP,
                "i": rinterface.INTSXP,
                # "u" -> special-cased below
                "f": rinterface.REALSXP,
                "c": rinterface.CPLXSXP,
                # "O" -> special-cased below
                "S": rinterface.STRSXP,
                "U": rinterface.STRSXP,
                # "V" -> special-cased below
            }
            # Most types map onto R arrays:
            if o.dtype.kind in kinds:
                # "F" means "use column-major order"
                # vec = rinterface.SexpVector(o.ravel("F"), kinds[o.dtype.kind])
                vec = rinterface.SexpVector(numpy.ravel(o, "F"), kinds[o.dtype.kind])
                res = vec
            # R does not support unsigned types:
            elif o.dtype.kind == "u":
                o = numpy.array(o, dtype=numpy.int64)
                return numpy2ri_vector(o)
                # raise(ValueError("Cannot convert numpy array of unsigned values -- R does not have unsigned integers."))
            # Array-of-PyObject is treated like a Python list:
            elif o.dtype.kind == "O":
                all_str = True
                all_bool = True
                for value in o:
                    if (
                            not isinstance(value, six.string_types) and
                            not type(value) is numpy.string_ and
                            not (type(value) is numpy.ma.core.MaskedArray and value.mask == True) and
                            not (type(value) is numpy.ma.core.MaskedConstant and value.mask == True)

                                ):
                        all_str = False
                        break
                    if not type(value) is bool or type(value) is numpy.bool_:
                        all_bool = False
                if (not all_str) and (not all_bool):
                    raise(ValueError("numpy2ri_vector currently does not handle object vectors: %s %s" % (value, type(value))))
                else:
                    #since we keep strings as objects
                    #we have to jump some hoops here
                    vec = rinterface.SexpVector(numpy.ravel(o,"F"), kinds['S'])
                    return vec
                    #res = ro.conversion.py2ri(list(o))
            # Record arrays map onto R data frames:
            elif o.dtype.kind == "V":
                raise(ValueError("numpy2ri_vector currently does not handle record arrays"))
            # It should be impossible to get here:
            else:
                raise(ValueError("Unknown numpy array type."))
        else:
            raise(ValueError("Unknown input to numpy2ri_vector."))
        return res

    ro.r("""
        dataframe_colname_setting = function(df, names)
        {
            colnames(df) = names
            df
        }
        """)

    def dataframe2ri(o):
        # print 'converting', o, type(o)
        # print repr(o)
        try:
            if isinstance(o, DataFrame):
                # print 'dataframe'
                dfConstructor = ro.r['data.frame']
                names = []
                parameters = []
                kw_params = {}
                for column_name in o.columns_ordered:
                    col = o.value_dict[column_name]
                    try:
                        names.append(str(column_name))
                        parameters.append(numpy2ri_vector(col))
                    except ValueError as e:
                        raise ValueError(str(e) + ' Offending column: %s, dtype: %s, content: %s' %( column_name, col.dtype, col[:10]))
                if not o.row_names is None:
                    kw_params['row.names'] = numpy2ri_vector(o.row_names)
                # print parameters
                ro.conversion.py2ri = previous_py2ri
                try:
                    res = dfConstructor(*parameters, **kw_params)
                    res = ro.r('dataframe_colname_setting')(res, names)
                except TypeError:
                    print(parameters.keys())
                    raise
                ro.conversion.py2ri = dataframe2ri
            elif isinstance(o, numpy.ndarray):
                # print 'numpy'
                res = numpy2ri_vector(o)
            else:
                # print 'old'
                #res =  ro.default_py2ri(o)
                res = previous_py2ri(o)
                # print 'not converting', o, type(o), res
            # print 'made it to return'

            return res
        except Exception as e:
            print(e)
            raise

    ro.conversion.py2ri = dataframe2ri
except ImportError: # guess we don't have rpy
    pass


def _replace_R_NAInteger_with_None(seq):
    import rpy2.rinterface
    res = []
    for value in seq:
        if type(value) == rpy2.rinterface.NAIntegerType:
            res.append(None)
        else:
            res.append(value)
    return res

__version__ = '0.1.7'
