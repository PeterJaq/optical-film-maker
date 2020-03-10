from common.TransferMatrix import OpticalModeling

import matplotlib.pyplot as plt

Demo = True  # set Demo to True to run an example simulation

if Demo:

    materials = ['Air', 'SiO2_Lemarchand', 'Cr_Rakic-LD', 'SiO2_Lemarchand', 'Al_McPeak']
    OM = OpticalModeling(Material=materials, 
                         WLrange=(250, 2000))

    thickness_list = [0, 90, 10, 80, 150]

    start = time.time()

    OM.RunSim(thickness_list)

    endtime = datetime.datetime.now()

    fig3 = plt.figure("Absorption", (9.6, 5.4))  # 16:9
    plt.clf()
    ax3 = fig3.add_subplot(111)
    ax3.set_ylabel('Absorption (%)', size=20)
    ax3.set_xlabel('Wavelength (nm)', size=20)
    ax3.tick_params(labelsize=18)

    ax3.plot(OM.WL, 100.0 * OM.Transmission,
             label="Transmission", linewidth=2)
    ax3.plot(OM.WL, 100.0 * OM.Reflection,
                label="Reflection", linewidth=2)
    ax3.plot(OM.WL, 100.0 * OM.total_Absorption,
                label="Absorption", linewidth=2)
    ax3.legend(loc='upper right', fontsize=14, frameon=False)

    ax3.set_xlim(xmin=OM.WL[0], xmax=OM.WL[-1])

    plt.show()

