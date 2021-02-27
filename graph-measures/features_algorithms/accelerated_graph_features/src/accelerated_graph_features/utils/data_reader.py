import csv

def __get_data_from_file(file_path, cast_to_float=False):
    from collections import OrderedDict
    """
    Master function to get data from .csv file
    :param file_path: the path to the data file
    :return: a list of dictionaries representing the data
    """
    data_list = []

    with open(file_path, encoding='utf8') as file:
        reader = csv.reader(file)

        # header row
        fields = reader.__next__()

        for row in reader:
            dictionary = OrderedDict()
            for i, field in enumerate(fields):
                if cast_to_float:
                    try:
                        try:
                            dictionary[field] = int(row[i])
                        except ValueError:
                            dictionary[field] = float(row[i])
                    except ValueError:
                        dictionary[field] = row[i]
                else:
                    dictionary[field] = row[i]
            data_list.append(dictionary)

    return data_list


def get_number_data(file):
    return __get_data_from_file(file, cast_to_float=True)
