
new_names = ['ab', 'ac', 'eg', 'be', 'cg', 'b1', 'b2']

perfect_states = ['ab', 'ac', 'c', 'e', 'f', 'h', 'i']

same_names = [
    # ['0', '0'],
    # ['a', 'a'],
    ['ab', 'a'],
    ['ac', 'a'],
    # ['b', 'b'],
    ['ba', 'b'],
    ['bd', 'be'],
    # ['be', 'be'],
    # ['bf', 'bf'],
    # ['c', 'c'],
    ['ca', 'c'],
    ['cd', 'c'],
    ['ce', 'c'],
    # ['cg', 'c'],
    ['d', 'e'],
    ['db', 'eb'],
    ['dc', 'e'],
    ['df', 'e'],
    ['dg', 'eg'],
    # ['e', 'e'],
    ['eb', 'eb'],
    ['ec', 'e'],
    ['ef', 'e'],
    # ['eg', 'eg'],
    # ['f', 'f'],
    ['fb', 'f'],
    ['fd', 'f'],
    ['fe', 'f'],
    ['fh', 'f'],
    # ['g', 'g'],
    ['gc', 'g'],
    ['gd', 'g'],
    ['ge', 'g'],
    ['gh', 'g'],
    # ['h', 'h'],
    ['hf', 'h'],
    ['hg', 'h']]

# same_names = [['e', 'd'], ['eb', 'db'], ['ec', 'dc'], ['ef', 'df'], ['eg', 'dg'], ['ce', 'cd'], ['fe', 'fd'],
#               ['ge', 'gd'], ['be', 'bd']]

states = {'ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h'}

connected = [['ab', 'ac'], ['ab', 'b', 'be', 'b1', 'b2'], ['ac', 'c'], ['c', 'e', 'cg'], ['eb', 'e'], ['e', 'f'],
             ['e', 'eg'], ['f', 'h'], ['h', 'g', 'i']]

pre_final_state = perfect_states[-2]
final_state = perfect_states[-1]
color_dict = {'b': '#9700fc', 'be': '#e0c1f5', 'b1': '#d108ba', 'b2': '#38045c',
              # 'bf': '#d108ba',
              # 'a': '#fc0000',
              'ac': '#fc0000', 'ab': '#802424',
              'c': '#fc8600', 'cg': '#8a4b03',
              'e': '#fcf400', 'eb': '#a6a103', 'eg': '#05f521',
              'f': '#30a103',
              'h': '#085cd1'}

allowed_transition_attempts = ['ab', 'ac',
                               'ba',
                               'ce', 'cd', 'ca',
                               'ec', 'ef',
                               'dc', 'df',
                               'fd', 'fe', 'fh',
                               'gh',
                               'hf', 'hg']