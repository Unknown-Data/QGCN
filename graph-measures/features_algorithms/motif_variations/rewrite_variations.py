import pickle


def rewrite_variations(level, directed):
    variations = pickle.load(open('{}_{}directed.pkl'.format(level, "" if directed else "un"), 'rb'))

    print('Reading ' + '{}_{}directed.pkl'.format(level, "" if directed else "un"))

    # Convert variations to idx -> [nums] format
    idx_to_nums = {}

    for num, idx in variations.items():
        if idx not in idx_to_nums.keys():
            idx_to_nums[idx] = []
        idx_to_nums[idx].append(num)

    # Write to file
    lines_to_write = ['\\begin{longtable}{|l|l|}\n',
                      '\t\\hline\n']

    # Add table rows
    for idx in idx_to_nums.keys():
        cell_str = ''
        nums_cell = [str(x) for x in idx_to_nums[idx]]
        line_size = 8
        sections = len(nums_cell) // line_size
        indices = [line_size * i for i in range(1, sections + 1)]
        prev_index = 0
        for e, i in enumerate(indices):
            section = nums_cell[prev_index:indices[e]]
            cell_str += ','.join(section) + ' \\newline '
            prev_index = indices[e]

        if sections == 0 or sections == 1:  # Less than 8 numbers
            cell_str = ','.join(nums_cell)

        row = '\t{} & {} \\\\ \\hline\n'.format(idx, cell_str)
        lines_to_write.append(row)
    # Postamble
    lines_to_write.append('\\end{longtable}')

    with open('{}_{}directed_rewritten.txt'.format(level, "" if directed else "un"), 'w+') as f:
        f.writelines(lines_to_write)


if __name__ == '__main__':
    rewrite_variations(3, False)
    rewrite_variations(3, True)
    rewrite_variations(4, False)
    rewrite_variations(4, True)
