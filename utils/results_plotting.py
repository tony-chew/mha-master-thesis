import matplotlib.pyplot as plt
from utils import rel_difference

FIG_SIZE = (9.9, 8.58)

NUMBER_LAYERS_SHALLOW = [ 4, 6, 8, 10 ]
NUMBER_LAYERS_DEEP = [ 10, 12, 14, 16 ]

SHALLOW_VAL_LOSS_256 = [0.042294063372537494, 0.04117628373205662, 0.040466672740876675, 0.03891198793426156]
SHALLOW_TEST_LOSS_256 = [0.029346848875894897, 0.02811384747690353, 0.02751811454133665, 0.026001292121459227]
DEEP_VAL_LOSS_512 = [0.024302403497345307, 0.023859524426774845, 0.023752347053959965, 0.02414328875463633]
DEEP_TEST_LOSS_512 = [0.016708042030146374, 0.016398102994092174, 0.016055478903113818, 0.016452040499167854]

FPS_256 = [748.6223450394416, 582.9727624093043, 483.11582904087885, 413.8488759822301]
FPS_512 = [343.1782005164969, 301.02982349087904, 266.14162293653857, 238.33823082820004]

PARAMS_256 = [269.738, 271.215, 272.692, 274.169]
PARAMS_512 = [10.280, 10.372, 16.275, 19.273]

GFLOPS_256 = [9.91, 14.75, 19.58, 24.42]
GFLOPS_512 = [24.93, 29.77, 34.61, 39.45]

SHALLOW_VAL_REL = rel_difference(SHALLOW_VAL_LOSS_256[0], SHALLOW_VAL_LOSS_256[3])
SHALLOW_TEST_REL = rel_difference(SHALLOW_TEST_LOSS_256[0], SHALLOW_TEST_LOSS_256[3])
SHALLOW_FPS_REL = rel_difference(FPS_256[0], FPS_256[3])
SHALLOW_PARAMS_REL = rel_difference(PARAMS_256[0], PARAMS_256[3])
SHALLOW_GFLOPS_REL = rel_difference(GFLOPS_256[0], GFLOPS_256[3])

DEEP_VAL_REL = rel_difference(DEEP_VAL_LOSS_512[0], DEEP_VAL_LOSS_512[3])
DEEP_TEST_REL = rel_difference(DEEP_TEST_LOSS_512[0], DEEP_TEST_LOSS_512[3])
DEEP_FPS_REL = rel_difference(FPS_512[0], FPS_512[3])
DEEP_PARAMS_REL = rel_difference(PARAMS_512[0], PARAMS_512[3])
DEEP_GFLOPS_REL = rel_difference(GFLOPS_512[0], GFLOPS_512[3])



def plot_layers_vs_results():
    """
    Plots the metrics against no of layers
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=FIG_SIZE)
    # fig.suptitle('Number of Layers Metrics', fontsize=20)

    # loss vs layers
    loss = ax[0, 0]
    # loss.set_title('Losses vs Number of Layers', fontsize=14)
    loss.set_xlabel("layer Number")
    loss.set_ylabel("Loss")
    loss.set_ylim([0.015, 0.045])
    loss.set_xlim([3.5, 16.5])
    loss.plot(NUMBER_LAYERS_SHALLOW, SHALLOW_VAL_LOSS_256, marker='o', color="red", label=f"rel%: {SHALLOW_VAL_REL}")
    loss.plot(NUMBER_LAYERS_SHALLOW, SHALLOW_TEST_LOSS_256, marker='o', color="blue", label=f"rel%: {SHALLOW_TEST_REL}")
    loss.legend(loc='lower left')

    loss1 = loss.twinx()
    loss1.set_ylim([0.015, 0.045])
    loss1.plot(NUMBER_LAYERS_DEEP, DEEP_VAL_LOSS_512, marker='x', color='red', label=f"rel%: {DEEP_VAL_REL}")
    loss1.plot(NUMBER_LAYERS_DEEP, DEEP_TEST_LOSS_512, marker='x', color='blue', label=f"rel%: {DEEP_TEST_REL}")
    loss1.legend(loc='upper right')
    loss1.yaxis.set_ticks([])

    # no of parameters vs layers
    met = ax[0, 1]
    # met.set_title('Parameters vs Number of Layers', fontsize=12)
    met.set_xlabel('Layer Number')
    met.set_ylabel('Number of Parameters (M)')
    met.set_ylim([268, 276])
    # met.plot(NUMBER_LAYERS_SHALLOW, PARAMS_256, marker='o', color="red", label='params @ latent size 256')
    met.plot(NUMBER_LAYERS_SHALLOW, PARAMS_256, marker='o',  label=f"rel%: {SHALLOW_PARAMS_REL}")
    met.legend(loc='upper left')
    met.tick_params(axis='y')

    met1 = met.twinx()
    # met1.plot(NUMBER_LAYERS_DEEP, PARAMS_512, marker='x', color='blue', label='params @ latent size 512\nwith optimised bottleneck')
    met1.plot(NUMBER_LAYERS_DEEP, PARAMS_512, marker='x',  label=f"rel%: {DEEP_PARAMS_REL}")
    met1.set_ylim([9, 21.5])
    met1.legend(loc='upper right')
    met1.tick_params(axis='y')

    # GFLOPs vs layers
    gfl = ax[1, 0]
    # gfl.set_title('GFLOPs vs Number of Layers', fontsize=12)
    gfl.set_xlabel('Layer Number')
    gfl.set_ylabel('GFLOPs')
    gfl.set_ylim([8, 46])
    # gfl.plot(NUMBER_LAYERS_SHALLOW, GFLOPS_256, marker='o', color="red", label='GFLOPs @ latent size 256')
    gfl.plot(NUMBER_LAYERS_SHALLOW, GFLOPS_256, marker='o',  label=f"rel%: {SHALLOW_GFLOPS_REL}")
    gfl.legend(loc='upper left')
    # gfl.plot(NUMBER_LAYERS_DEEP, GFLOPS_512, marker='x', color='blue', label='GFLOPs @ latent size 512\nwith optimised bottleneck')

    gfl1 = gfl.twinx()
    gfl1.plot(NUMBER_LAYERS_DEEP, GFLOPS_512, marker='x', label=f"rel%: {DEEP_GFLOPS_REL}")
    gfl1.set_ylim([8, 46])
    gfl1.axis('off')
    gfl1.legend(loc='upper right')

    # FPS vs layers
    fps = ax[1, 1]
    # fps.set_title('FPS vs Number of Layers', fontsize=12)
    fps.set_xlabel('Layer Number')
    fps.set_ylabel('FPS')
    fps.set_ylim([225, 840])
    # fps.plot(NUMBER_LAYERS_SHALLOW, FPS_256, marker='o', color="red", label='FPS @ latent size 256')
    fps.plot(NUMBER_LAYERS_SHALLOW, FPS_256, marker='o', label=f"rel%: {SHALLOW_FPS_REL}")
    fps.legend(loc="upper left")
    # fps.plot(NUMBER_LAYERS_DEEP, FPS_512, marker='x', color='blue', label='FPS @ latent size 512\nwith optimised bottleneck')

    fps1 = fps.twinx()
    fps1.plot(NUMBER_LAYERS_DEEP, FPS_512, marker='x',  label=f"rel%: {DEEP_FPS_REL}")
    fps1.legend(loc='upper right')
    fps1.set_ylim([225, 840])
    fps1.axis('off')

    plt.show()


COMP = [ 'W/O Compression\n2D Bottleneck', 'W/ Compression\n2D Bottleneck', 'W/ Compression\n3D Bottleneck']
COMP_VAL_LOSS = [0.031344452063026634, 0.031766971006341606, 0.024302403497345307]
COMP_VAL_REL = rel_difference(COMP_VAL_LOSS[0], COMP_VAL_LOSS[2])
COMP_TEST_LOSS = [0.020537930299512674, 0.020850007290661825, 0.016708042030146374]
COMP_TEST_REL = rel_difference(COMP_TEST_LOSS[0], COMP_TEST_LOSS[2])
COMP_FPS = [ 433.904987318664, 385.49192957076133, 343.1782005164969 ]
COMP_FPS_REL = rel_difference(COMP_FPS[0], COMP_FPS[2])
COMP_PARAMS = [ 542.605, 72.471, 10.280 ]
COMP_PARAMS_REL = rel_difference(COMP_PARAMS[0], COMP_PARAMS[2])
COMP_GFLOPS = [24.69, 24.39, 24.93]
COMP_GFLOPS_REL = rel_difference(COMP_GFLOPS[0], COMP_GFLOPS[2])


def plot_compression_strategies():
    """
    Plots the metrics against 3D vs 2D bottlenecks
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=FIG_SIZE)
    # fig.suptitle('Bottleneck Compression Strategies', fontsize=20)

    # loss vs 3D and 2D bottlenecks
    loss = ax[0, 0]
    # loss.set_title('Losses vs Bottleneck Compression Strategies', fontsize=12)
    loss.set_ylabel("Loss")
    loss.set_ylim([0, 0.035])
    # loss.bar(COMP, COMP_VAL_LOSS, width=-0.2, color="red", align="edge", label=f'val loss, rel%: {COMP_VAL_REL}')
    loss.bar(COMP, COMP_VAL_LOSS, width=-0.2, color="red", align="edge", label=f"rel%: {COMP_VAL_REL}")
    # loss.bar(COMP, COMP_TEST_LOSS, width=0.2, color="blue", align="edge", label=f'test loss, rel%: {COMP_TEST_REL}')
    loss.bar(COMP, COMP_TEST_LOSS, width=0.2, color="blue", align="edge", label=f"rel%: {COMP_TEST_REL}")
    loss.legend(loc='upper right')

    # no of parameters vs 3D and 2D bottlenecks
    met = ax[0, 1]
    # met.set_title('Parameters vs Bottleneck Compression Strategies', fontsize=12)
    met.set_ylabel('Number of Parameters (M)')
    met.set_ylim([0, 560])
    # met.bar(COMP, COMP_PARAMS, width=0.2, label=f"params, rel%: {COMP_PARAMS_REL}")
    met.bar(COMP, COMP_PARAMS, width=0.2, label=f"rel%: {COMP_PARAMS_REL}")
    met.legend(loc='upper right')

    # GFLOPs vs 3D and 2D bottlenecks
    gfl = ax[1, 0]
    # gfl.set_title('GFLOPs vs Bottleneck Compression Strategies', fontsize=12)
    gfl.set_ylabel('GFLOPs')
    gfl.set_ylim([0, 30])
    # gfl.bar(COMP, COMP_GFLOPS, width=0.2, label=f'GFLOPs, rel%: {COMP_GFLOPS_REL}')
    gfl.bar(COMP, COMP_GFLOPS, width=0.2, label=f"rel%: {COMP_GFLOPS_REL}")
    gfl.legend(loc='upper right')

    # FPS vs 3D and 2D bottlenecks
    fps = ax[1, 1]
    # fps.set_title('FPS vs Bottleneck Compression Strategies', fontsize=12)
    fps.set_ylabel('FPS')
    fps.set_ylim([0, 450])
    # fps.bar(COMP, COMP_FPS, width=0.2, label=f'FPS, rel%: {COMP_FPS_REL}')
    fps.bar(COMP, COMP_FPS, width=0.2, label=f"rel%: {COMP_FPS_REL}")
    fps.legend(loc='upper right')

    plt.show()


RES = ['  ', 'W/O Residual', '    ', 'W/ Residual', '      ']
RES_VAL_LOSS = [ 0, 0.024302403497345307, 0, 0.024624466676922405, 0 ]
RES_VAL_REL = rel_difference(RES_VAL_LOSS[1], RES_VAL_LOSS[3])
RES_TEST_LOSS = [ 0, 0.016708042030146374, 0, 0.017001547457375487, 0 ]
RES_TEST_REL = rel_difference(RES_TEST_LOSS[1], RES_TEST_LOSS[3])
RES_FPS = [ 0, 343.1782005164969, 0, 156.10711313883297, 0]
RES_FPS_REL = rel_difference(RES_FPS[1], RES_FPS[3])
RES_PARAMS = [0, 10.280, 0, 11.768, 0]
RES_PARAMS_REL = rel_difference(RES_PARAMS[1], RES_PARAMS[3])
RES_GFLOPS = [0, 24.93, 0, 27.76, 0]
RES_GFLOPS_REL = rel_difference(RES_GFLOPS[1], RES_GFLOPS[3])

def plot_residuals_vs_results():
    """
    Plots the metrics against residual or no residual modules
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=FIG_SIZE)
    # fig.suptitle('Residual Modules Metrics', fontsize=20)

    # loss vs Residual Modules Metrics
    loss = ax[0, 0]
    # loss.set_title('Losses vs Residual Modules Metrics', fontsize=14)
    loss.set_ylabel("Loss")
    loss.set_ylim([0, 0.032])
    # loss.bar(RES, RES_VAL_LOSS, width=-0.4, color='red', align='edge', label=f'val loss, rel %: {RES_VAL_REL}')
    loss.bar(RES, RES_VAL_LOSS, width=-0.4, color='red', align='edge', label=f'rel%: {RES_VAL_REL}')
    # loss.bar(RES, RES_TEST_LOSS, width=0.4, color='blue', align='edge', label=f'test loss, rel %: {RES_TEST_REL}')
    loss.bar(RES, RES_TEST_LOSS, width=0.4, color='blue', align='edge', label=f'rel%: {RES_TEST_REL}')
    loss.legend(loc='upper right')

    # no of parameters vs Residual Modules Metrics
    met = ax[0, 1]
    # met.set_title('Parameters vs Residual Modules Metrics', fontsize=14)
    met.set_ylabel('Number of Parameters (M)')
    met.set_ylim([0, 14])
    # met.bar(RES, RES_PARAMS, width=0.4, label=f"params, rel %: {RES_PARAMS_REL}")
    met.bar(RES, RES_PARAMS, width=0.4, label=f"rel%: {RES_PARAMS_REL}")
    met.legend(loc='upper right')

    # GFLOPs vs Residual Modules Metrics
    gfl = ax[1, 0]
    # gfl.set_title('GFLOPs vs Residual Modules Metrics', fontsize=14)
    gfl.set_ylabel('GFLOPs')
    gfl.set_ylim([0, 33])
    # gfl.bar(RES, RES_GFLOPS, width=0.4, label=f'GFLOPs, rel %: {RES_GFLOPS_REL}')
    gfl.bar(RES, RES_GFLOPS, width=0.4, label=f'rel%: {RES_GFLOPS_REL}')
    gfl.legend(loc='upper right')

    # FPS vs Residual Modules Metrics
    fps = ax[1, 1]
    # fps.set_title('FPS vs Residual Modules Metrics', fontsize=14)
    fps.set_ylabel('FPS')
    fps.set_ylim([0, 360])
    # fps.bar(RES, RES_FPS, width=0.4, label=f'FPS, rel %: {RES_FPS_REL}')
    fps.bar(RES, RES_FPS, width=0.4, label=f'rel%: {RES_FPS_REL}')
    fps.legend(loc='upper right')

    plt.show()


LATENT_SIZE = [ '256', '512', '1024' ]
LATENT_VAL_LOSS = [ 0.031766971006341606, 0.024302403497345307, 0.017960602581939277 ]
LATENT_VAL_REL = rel_difference(LATENT_VAL_LOSS[0], LATENT_VAL_LOSS[2])
LATENT_TEST_LOSS = [ 0.020850007290661825, 0.016708042030146374, 0.012360253095871113 ]
LATENT_TEST_REL = rel_difference(LATENT_TEST_LOSS[0], LATENT_TEST_LOSS[2])
LATENT_PARAMS = [ 10.28, 10.280198, 10.280265 ]
LATENT_PARAMS_REL = rel_difference(LATENT_PARAMS[0], LATENT_PARAMS[2])
LATENT_FPS = [ 385.49192957076133, 343.1782005164969, 343.1513504074701]
LATENT_FPS_REL = rel_difference(LATENT_FPS[0], LATENT_FPS[2])
LATENT_GFLOPS = [ 25.8, 24.93, 24.93 ]
LATENT_GFLOPS_REL = rel_difference(LATENT_GFLOPS[0], LATENT_GFLOPS[2])

def plot_latent_vs_results():
    """
    Plots the metrics against residual or no residual modules
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=FIG_SIZE)
    # fig.suptitle('Latent Size Metrics', fontsize=20)

    # loss vs Latent Size Metrics
    loss = ax[0, 0]
    # loss.set_title('Losses vs Latent Size Metrics', fontsize=12)
    loss.set_ylabel("Loss")
    loss.set_ylim([0, 0.033])
    # loss.bar(LATENT_SIZE, LATENT_VAL_LOSS, width=-0.2, color="red", align='edge', label=f'val loss, rel %: {LATENT_VAL_REL}')
    loss.bar(LATENT_SIZE, LATENT_VAL_LOSS, width=-0.2, color="red", align='edge', label=f'rel%: {LATENT_VAL_REL}')
    # loss.bar(LATENT_SIZE, LATENT_TEST_LOSS, width=0.2, color="blue", align='edge', label=f'test loss, rel %: {LATENT_TEST_REL}')
    loss.bar(LATENT_SIZE, LATENT_TEST_LOSS, width=0.2, color="blue", align='edge', label=f'rel%: {LATENT_TEST_REL}')
    loss.legend(loc='upper right')

    # no of parameters vs Latent Size Metrics
    met = ax[0, 1]
    # met.set_title('Parameters vs Latent Size Metrics', fontsize=12)
    met.set_ylabel('Number of Parameters (M)')
    met.set_ylim([0, 12])
    # met.bar(LATENT_SIZE, LATENT_PARAMS, width=0.2, label=f"params, rel %: {LATENT_PARAMS_REL}")
    met.bar(LATENT_SIZE, LATENT_PARAMS, width=0.2, label=f"rel%: {LATENT_PARAMS_REL}")
    met.legend(loc='upper right')

    # GFLOPs vs Latent Size Metrics
    gfl = ax[1, 0]
    # gfl.set_title('GFLOPs vs Latent Size Metrics', fontsize=12)
    gfl.set_ylabel('GFLOPs')
    gfl.set_ylim([0, 30])
    # gfl.bar(LATENT_SIZE, LATENT_GFLOPS, width=0.2, label=f'GFLOPs, rel %: {LATENT_GFLOPS_REL}')
    gfl.bar(LATENT_SIZE, LATENT_GFLOPS, width=0.2, label=f'rel%: {LATENT_GFLOPS_REL}')
    gfl.legend(loc='upper right')

    # FPS vs Latent Size Metrics
    fps = ax[1, 1]
    # fps.set_title('FPS vs Latent Size Metrics', fontsize=12)
    fps.set_ylabel('FPS')
    fps.set_ylim([0, 420])
    # fps.bar(LATENT_SIZE, LATENT_FPS, width=0.2, label=f'FPS, rel %: {LATENT_FPS_REL}')
    fps.bar(LATENT_SIZE, LATENT_FPS, width=0.2, label=f'rel%: {LATENT_FPS_REL}')
    fps.legend(loc='upper right')

    plt.show()


AE_MODEL = ['ConvAE10', 'ResAE10', 'ERFNetAE', 'FastSCNNAE', 'LEDNetAE', 'DABNetAE', 'EDANetAE' ]
AE_MODEL_VAL_LOSS = [ 0.024302403497345307, 0.024624466676922405, 0.025740477838553488, 0.0260669655399397, 0.027141717337071894, 0.024580048888714776, 0.025290382759911672 ]
AE_MODEL_TEST_LOSS = [ 0.016708042030146374, 0.017001547457375487, 0.019380599560490885, 0.018102891737686807, 0.01797612709649762, 0.01730293108883207, 0.0176462691817738 ]
AE_MODEL_PARAMS = [ 10.280, 11.76, 5.148, 2.524, 3.229, 3.513, 3.221]
AE_MODEL_FPS = [ 343.1782005164969, 156.10711313883297, 75.09300844324592, 136.651482360116, 44.392613579263326, 68.44562851414007, 76.95612116694679 ]
AE_MODEL_GFLOPS = [ 24.93, 27.76, 47.84, 6.65, 28.52, 34.17, 38.12 ]

def plot_autoencoder_results():
    """
    Plots the chosen autoencoder metrics
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=FIG_SIZE)
    # fig.suptitle('Autoencoder Metrics', fontsize=20)

    # loss vs Autoencoder Metrics
    loss = ax[0, 0]
    # loss.set_title('Losses vs Autoencoder Metrics', fontsize=12)
    loss.set_ylabel("Loss")
    loss.set_ylim([0.0, 0.035])
    loss.bar(AE_MODEL, AE_MODEL_VAL_LOSS, width=-0.4, color="red", align="edge", label="val los")
    loss.bar(AE_MODEL, AE_MODEL_TEST_LOSS, width=0.4, color="blue", align="edge", label="test loss")
    loss.legend(loc='upper right')

    # no of parameters vs Autoencoder Metrics
    met = ax[0, 1]
    # met.set_title('Parameters vs Autoencoder Metrics', fontsize=12)
    met.set_ylabel('Number of Parameters (M)')
    met.set_ylim([0, 13])
    met.bar(AE_MODEL, AE_MODEL_PARAMS, width=0.4, label="params")
    met.legend(loc='upper right')

    # GFLOPs vs Autoencoder Metrics
    gfl = ax[1, 0]
    # gfl.set_title('GFLOPs vs Autoencoder Metrics', fontsize=12)
    gfl.set_ylabel('GFLOPs')
    gfl.set_ylim([0, 55])
    gfl.bar(AE_MODEL, AE_MODEL_GFLOPS, width=0.4, label='GFLOPs')
    gfl.legend(loc='upper right')

    # FPS vs Autoencoder Metrics
    fps = ax[1, 1]
    # fps.set_title('FPS vs Autoencoder Metrics', fontsize=12)
    fps.set_ylabel('FPS')
    fps.set_ylim([0, 360])
    fps.bar(AE_MODEL, AE_MODEL_FPS, width=0.4, label='FPS')
    fps.legend(loc='upper right')

    plt.show()


ERFNET = ['12', '14', '16']
ERF_VAL_LOSS = [0.02433187576631705, 0.02368524044752121, 0.0232167478534393]
ERF_VAL_REL = rel_difference(ERF_VAL_LOSS[0], ERF_VAL_LOSS[2])
ERF_TEST_LOSS = [0.01705011840818114, 0.016672959517626488, 0.01636544561868564]
ERF_TEST_REL = rel_difference(ERF_TEST_LOSS[0], ERF_TEST_LOSS[2])
ERF_FPS = [119.92105800495911, 103.23355298211617, 77.58408747100746]
ERF_FPS_REL = rel_difference(ERF_FPS[0], ERF_FPS[2])
ERF_PARAMS = [4.087, 4.877, 6.458]
ERF_PARAMS_REL = rel_difference(ERF_PARAMS[0], ERF_PARAMS[2])
ERF_GFLOPS = [32.75, 39.21, 52.13]
ERF_GFLOPS_REL = rel_difference(ERF_GFLOPS[0], ERF_GFLOPS[2])


def main():
    plot_layers_vs_results()
    # plot_compression_strategies()
    # plot_residuals_vs_results()
    # plot_latent_vs_results()
    # plot_autoencoder_results()

if __name__ == '__main__':
    main()