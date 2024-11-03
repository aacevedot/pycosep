import numpy as np


def _half_kernel():
    embedding = np.array([[0.890083735825258, -3.89971164872729],
                          [1.73553495647023, -3.60387547160968],
                          [2.49395920743493, -3.12732592987212],
                          [3.12732592987212, -2.49395920743493],
                          [3.60387547160968, -1.73553495647023],
                          [3.89971164872729, -0.890083735825258],
                          [4, 0],
                          [3.89971164872729, 0.890083735825258],
                          [3.60387547160968, 1.73553495647023],
                          [3.12732592987212, 2.49395920743493],
                          [2.49395920743493, 3.12732592987212],
                          [1.73553495647023, 3.60387547160968],
                          [0.890083735825258, 3.89971164872729],
                          [2.22044604925031e-16, 4],
                          [-0.890083735825257, 3.89971164872729],
                          [1.33512560373789, -5.84956747309094],
                          [2.27972339226216, -5.5500325453796],
                          [3.16213626944087, -5.09910719768535],
                          [3.95829431226328, -4.50909149801866],
                          [4.64648040475032, -3.79607953661028],
                          [5.20792262426618, -2.97952042108398],
                          [5.62730629063311, -2.08168775550057],
                          [5.89319171068425, -1.12707207450208],
                          [5.99832622327216, -0.141712805368787],
                          [5.93984203295867, 0.847512019677251],
                          [5.71933443599862, 1.81361892612544],
                          [5.34281830481667, 2.73025503602062],
                          [4.82056401796717, 3.57241690577685],
                          [4.16681731098005, 4.31713255493933],
                          [3.39941068983145, 4.94408808192771],
                          [2.53927700666655, 5.43618177431684],
                          [1.60987846616091, 5.77999059897085],
                          [0.636566637738349, 5.9661363473959],
                          [-0.354109069156336, 5.98954144882071],
                          [-1.33512560373789, 5.84956747309094]])

    communities = np.array(
        ['Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1',
         'Group1', 'Group1', 'Group1', 'Group1', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2',
         'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2',
         'Group2', 'Group2'])

    return embedding, communities


def _parallel_lines():
    embedding = np.array([[1, 1],
                          [2, 2],
                          [3, 3],
                          [4, 4],
                          [5, 5],
                          [6, 6],
                          [2, 1],
                          [3, 2],
                          [4, 3],
                          [5, 4],
                          [6, 5],
                          [7, 6]])

    communities = np.array(
        ['Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2',
         'Group2'])

    return embedding, communities


def _circles():
    embedding = np.array([[2.8, 0],
                          [2.64828827676178, 0.909158513773114],
                          [2.2095934263099, 1.71979559553107],
                          [1.5314548427428, 2.34406613913508],
                          [0.687359363994238, 2.71432074463012],
                          [-0.23122216732253, 2.79043658041868],
                          [-1.12474718902831, 2.56416531463416],
                          [-1.89638840055207, 2.06002694988477],
                          [-2.46252650337817, 1.33265270050381],
                          [-2.76181164952762, 0.460864852786055],
                          [-2.76181164952762, -0.460864852786055],
                          [-2.46252650337817, -1.3326527005038],
                          [-1.89638840055208, -2.06002694988477],
                          [-1.12474718902831, -2.56416531463416],
                          [-0.231222167322532, -2.79043658041868],
                          [0.687359363994237, -2.71432074463013],
                          [1.53145484274279, -2.34406613913508],
                          [2.2095934263099, -1.71979559553107],
                          [2.64828827676178, -0.909158513773114],
                          [2.8, -6.85802207522518e-16],
                          [4, 0],
                          [3.78326896680254, 1.29879787681873],
                          [3.15656203758557, 2.45685085075867],
                          [2.18779263248971, 3.34866591305011],
                          [0.981941948563197, 3.87760106375732],
                          [-0.330317381889329, 3.98633797202668],
                          [-1.60678169861188, 3.66309330662023],
                          [-2.70912628650296, 2.94289564269253],
                          [-3.51789500482596, 1.90378957214829],
                          [-3.94544521361089, 0.658378361122936],
                          [-3.94544521361089, -0.658378361122935],
                          [-3.51789500482596, -1.90378957214829],
                          [-2.70912628650296, -2.94289564269253],
                          [-1.60678169861188, -3.66309330662023],
                          [-0.330317381889331, -3.98633797202668],
                          [0.981941948563195, -3.87760106375732],
                          [2.18779263248971, -3.34866591305012],
                          [3.15656203758558, -2.45685085075867],
                          [3.78326896680254, -1.29879787681873],
                          [4, -9.79717439317883e-16]])

    communities = np.array(
        ['Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1',
         'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group2', 'Group2',
         'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2',
         'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2'])

    return embedding, communities


def _rhombus():
    embedding = np.array([[11, 15],
                          [11, 13],
                          [11, 11],
                          [10, 12],
                          [10, 14],
                          [9, 13],
                          [12, 15],
                          [13, 14],
                          [12, 13],
                          [13, 12],
                          [12, 11],
                          [14, 13],
                          [12, 9],
                          [11, 9],
                          [10, 10],
                          [9, 11],
                          [8, 12],
                          [13, 10],
                          [14, 11],
                          [15, 12]])

    communities = np.array(
        ['Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2',
         'Group2', 'Group2', 'Group1', 'Group1', 'Group1', 'Group1', 'Group2', 'Group2', 'Group2'])

    return embedding, communities


def _spirals():
    embedding = np.array([[0, 0],
                          [0.213707292172779, 0.0780092511961152],
                          [0.348003195214442, 0.293119047693149],
                          [0.339592240186104, 0.592016351467916],
                          [0.154670100039219, 0.896759254289499],
                          [-0.202754888301297, 1.1192840145691],
                          [-0.689114932778805, 1.17828078547571],
                          [-1.22659327207034, 1.0156402881492],
                          [-1.71484466001696, 0.609678433283747],
                          [-2.04742763822007, -0.0172138389842368],
                          [-2.13043896358656, -0.798031842993885],
                          [-1.90039614549161, -1.62818940550558],
                          [-1.33841204226011, -2.37940185868909],
                          [-0.478157409866028, -2.91859071152332],
                          [0.594041896610901, -3.1291115712085],
                          [1.74749169108778, -2.93111395881825],
                          [2.82305979777731, -2.29781056185525],
                          [3.65480826590013, -1.26488449650872],
                          [4.09442111599008, 0.0688529224834121],
                          [4.03494339706133, 1.55023805672263],
                          [3.43025525726204, 2.98928902417049],
                          [2.30713091126124, 4.18349772419],
                          [0.767636179499515, 4.94578201055439],
                          [-1.0191104783898, 5.13229676488374],
                          [-2.8353160514771, 4.66611003816201],
                          [-4.4410598255077, 3.55311748697697],
                          [-5.60577327660553, 1.88746681329642],
                          [-6.14054632400641, -0.154909950394924],
                          [-5.92681535330728, -2.33447205333478],
                          [-4.93725338505321, -4.37613245478021],
                          [-3.24553953357265, -6.00392356180665],
                          [-1.02304035252633, -6.97790403252315],
                          [0.610896403996775, 0.610896403996775],
                          [0.526766383186186, 1.13241478918944],
                          [0.13936576282195, 1.62798357048857],
                          [-0.528002720606541, 1.94867228980591],
                          [-1.38619061156941, 1.96402479587702],
                          [-2.29201095177198, 1.5889810734486],
                          [-3.07034477260803, 0.804279352300409],
                          [-3.54329106797934, -0.33335859250014],
                          [-3.5618658291108, -1.6934457781182],
                          [-3.03517845551987, -3.08664798374533],
                          [-1.95220045853987, -4.29070211579474],
                          [-0.392183682306755, -5.08383324672817],
                          [1.47863632344276, -5.28083425119552],
                          [3.42441753035886, -4.76631908161556],
                          [5.16939922845775, -3.51980849922546],
                          [6.43616833865968, -1.62826122450658],
                          [6.98727872922341, 0.716687312108089],
                          [6.66420918453467, 3.23738751655652],
                          [5.417703454949, 5.60303118911751],
                          [3.32446533343765, 7.47281451149156],
                          [0.586884248685325, 8.54380480813029],
                          [-2.48526982986914, 8.59691368096278],
                          [-5.50971868739901, 7.53428153146049]])

    communities = np.array(
        ['Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1',
         'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1',
         'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group1', 'Group2',
         'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2',
         'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2', 'Group2'])

    return embedding, communities
