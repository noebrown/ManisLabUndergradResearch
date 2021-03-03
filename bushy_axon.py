"""
Example model of a cell, and axon, and a postsynaptic cell with cnmodel.
Output plots are:
top : voltage at each node of ranvier
middle: postsynaptic current
bottom: presynaptic signal

"""
import argparse
import numpy as np
import sys

import pyqtgraph as pg
import matplotlib.pyplot as mpl
import cnmodel.cells
import cnmodel.util as CU
from neuron import h


class Model():
    def __init__(self):
        self.temp = 34.
        self.dt = 0.005
        self.current_traces = []
        self.axon_voltage_traces = []
        self.post_cell_current_traces = []
        self.sgc_voltage_traces = []
        self.calyx_ca = []
        self.initdelay = 0.
        
        # first, make an SGC cell (this is an Hodgkin-Huxley style model
        # of an SGC cell - it could be replaced with a Bushy cell)
        # The precise model here is not critical, as long as it can generate
        # a single spike in response to short current pulses.
        self.cell = cnmodel.cells.SGC.create(model='I',
                species='mouse',
                modelType='sgc-bm', ttx=False)

        # Next add an axon to the cell using the add_axon tool in the cell class.
        self.cell.add_axon(internodeDiameter=4.0, internodeLength=250., internodeELeak=-65.,
                            nodeDiameter=2.0, nodeLength=1.0, nodeELeak=-65., 
                            nodes=10, natype='nacncoop')

        # Now make a postsynaptic cell (this could be a VCN bushy cell, or you can think of it as an MNTB neuron).
        self.post_cell = cnmodel.cells.Bushy.create(
                species='mouse',
                modelName='XM13', modelType='II', ttx=False)

        # Next, we connect the SGC cell (self.cell) to the postsynaptic cell (self.post_cell) by a synapse. This synapse mimics a calyx of Held or an endbulb of Held.
        synapsetype='multisite'  # stochastic, multiple release site synapse
        sgc_synapses = []  # make an array to hold the synapses (we might want more than one someday)
        n_synapses = 100  # just one for now

        h.topology()  # print out the "topology" of our cell.
        
        for i in range(n_synapses):  # connect synapses between the cells.
            pre_cell =  cnmodel.cells.cell_from_section(self.cell.axnode[-1])
            post_cell = cnmodel.cells.cell_from_section(self.post_cell.soma)
            opts = {'spike_source': 'cai', 'spike_section': self.cell.axnode[-1]}
            sgc_synapses.append(pre_cell.connect(post_cell, type=synapsetype, pre_opts=opts))
            sgc_synapses[-1].terminal.netcon.threshold = 0.001  # neuron parameters for synaptic connections
            sgc_synapses[-1].terminal.netcon.delay = 0.0
            sgc_synapses[-1].terminal.relsite.latency = 0.0
            print(sgc_synapses[-1].terminal.relsite.latency)
        if len(sgc_synapses) == 0:
            raise ValueError('No synapses created for this cell combination!')
        
        # call a routine to adjust the conductances - 
        self.zero_K()

        self.cell.set_temperature(float(self.temp))
        """
        Get the resting potential:
        The cnmodel.cell base class uses the Brent method:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
        to find the zero of the current-voltage relationship over an interval.
        There is no analytical solution.
        """
        V0 = self.cell.find_i0(showinfo=True)
        # (Newton-Raphson or other search method)
        print('Currents at nominal Vrest= %.2f I = 0: I = %g ' % (V0, self.cell.i_currents(V=V0)))
        resting_meas = self.cell.compute_rmrintau(auto_initialize=False, vrange=None)
        print('    From Inst: Rin = {:7.1f}  Tau = {:7.1f}  Vm = {:7.1f}'.format(resting_meas['Rin'], resting_meas['tau'], resting_meas['v']))

        # note that the run is called from the top level main routine (see end
        # of source file)
        

    def zero_K(self):
        """
        Adjust conductances to do 'experiments' on the axon
        """
        # print('zero_K axnode: ', self.cell.axnode)
        i = 0
        for node in self.cell.axnode:
            for seg in node:
                if i == 0:
                    print(f"KLT:  {seg.klt.gbar:.6f} mho/cm2")
                    print(f"KCNQ: {seg.kcnq.gbar:.6f} mho/cm2")
                    print(f"KHT: {seg.kht.gbar:.6f} mho/cm2")
                i = 1
                # seg.klt.gbar = 0e-3
                seg.kcnq.gbar = 0e-3
                # seg.kcnq.phi_m = seg.kcnq.phi_m - 20.
                # seg.kht.gbar = 0e-3
                # seg.kht.vshift = -20.
                pass



    def run(self):
        """
        Perform a simulation run on the little network we built,
        and plot the results
        """
        sites = [x for x in self.cell.all_sections['axonnode']]
        ptype = 'pulses'
        self.default_durs=[10., 100., 25.]
        stimdict = {  # set up the stimulus parameters
                'NP': 20,
                'Sfreq': 500.0,
                'delay': self.default_durs[0],
                'dur': 0.5,
                'amp': 2.0,
                'PT': 0.0,
                'dt': self.dt,
                }
        istim = h.iStim(0.5, sec=self.cell.soma)
        istim.delay = 0.
        istim.dur = 1e9 # these actually do not matter...
        istim.iMax = 0.0
        self.run_one(istim, stimdict, sites=sites) #do one simulation

    def run_one(self, istim, stim, initflag=True, sites=None):
        """
        Perform one run in current-clamp for the selected cell
        and add the data to the traces
        Monitor the presyanptic cell body and axon, as well as
        the current in a voltage-clamped target cell
        
        Parameters
        ----------
        istim : Stimulus electrode instance
        stim : waveform information
        initflag : boolean (default: True)
            If true, force initialziation of the cell and computation of 
            point Rin, tau and Vm
        sites : list of sections to monitor in addition to the soma
        """
        h.dt = self.dt  # establish this before we do any play/record setup

        # print('iv_curve:run_one')
        (secmd, maxt, tstims) = cnmodel.util.stim.make_pulse(stim)
        # print('maxt, dt*lencmd: ', maxt, len(secmd)*self.dt)# secmd = np.append(secmd, [0.])
        # print('stim: ', stim, self.tend)

        # connect current command vector
        playvector = h.Vector(secmd)  # sets up to "play" the array in secmd to # the stimulating electrode in the cell 
        # Python-> neuron call to set up the stimulus
        playvector.play(istim._ref_i, h.dt, 0, sec=self.cell.soma)

        self.r = {}  # results stored in a dict
        # Connect recording vectors
        self.r['v_soma'] = h.Vector()  # neuron arrays are "Vector"
        # set up an electrode to record from the cell that is directly stimulated
        self.r['v_soma'].record(self.cell.soma(0.5)._ref_v, sec=self.cell.soma)
        # self['q10'] = self.cell.soma(0.5).ihpyr_adj._ref_q10
        # self['ih_ntau'] = self.cell.soma(0.5).ihpyr_adj._ref_kh_n_tau
        # also record the current injection.
        self.r['i_inj'] = h.Vector()
        self.r['i_inj'].record(istim._ref_i, sec=self.cell.soma)
        self.r['time'] = h.Vector()
        self.r['time'].record(h._ref_t)

        # possibly record from all the axon nodes in the cell 
        # sites is set up in the main initialization routine
        if sites is not None:
            recvec = [h.Vector() for x in sites]
            for i, r in enumerate(recvec):
                r.record(sites[i](0.5)._ref_v, h.dt, 0, sec=sites[i])
        
        # for fun, monitor the presynaptic calcium concentration
        self.r['calyxca'] = h.Vector()
        self.r['calyxca'].record(sites[-1](1)._ref_cai, h.dt, 0, sec=sites[-1])
        h.celsius = self.cell.status['temperature']
        # print("iv_curve:run_one:calling cell_initialize")
        # set up an electrode in the postaynaptic cell.
        # This is a voltage clamp electrode, but we are just holding
        # the voltage constant. The parameters are required to be
        # set for this to work.
        clampV = -65.
        vccontrol = h.SEClamp(0.5, sec=self.post_cell.soma)
        vccontrol.dur1 = 10.0
        vccontrol.amp1 = clampV
        vccontrol.dur2 = 100.0
        vccontrol.amp2 = clampV
        vccontrol.dur3 = 25.0
        vccontrol.amp3 = clampV
        vccontrol.rs = 1e-9  # close to perfect - no series resistance
        
        # record the current through the postsynaptic cell membrane
        # this is the synaptic current
        self.r['i_post'] = h.Vector()
        self.r['i_post'].record(vccontrol._ref_i, sec=self.post_cell.soma)

        # print("iv_curve:run_one:calling custom_init")
        CU.custom_init()  # using our own initialization

        # now we can actually set up the run
        h.t = 0.
        h.tstop = np.sum(self.default_durs)
        print('stop: ', h.tstop, '  dt: ', h.dt)
        # and here it goes...
        while h.t < h.tstop:
            h.fadvance()
        # done! so now save the data from this run
        print('stop: ', h.tstop)
        # store the results in lists
        self.sgc_voltage_traces.append(self.r['v_soma'])
        self.current_traces.append(self.r['i_inj'])
        self.calyx_ca.append(self.r['calyxca'])
        self.post_cell_current_traces.append(self.r['i_post'])
        print('sgc  trace len: ', len(self.sgc_voltage_traces))
        print('post trace len: ', len(self.post_cell_current_traces))
        self.time_values = np.array(self.r['time'])
        if sites is not None:
            # for i, r in enumerate(recvec):
            #     print('r2: ', np.array(recvec[i]))
            self.axon_voltage_traces.append(np.array(recvec.copy()))
            # print('# axon site traces: ', len(self.axon_voltage_traces))
        # self.mon_q10 = np.array(self['q10'])
        # self.mon_ih_ntau = np.array(self['ih_ntau'])

    def show_pg(self, cell=None, rmponly=False):
        print("rmpvalue : ")
        print(rmponly)
        """
        Plot results from run_iv()
        Using pyqtgraph.
        
        Parameters
        ----------
        cell : cell object (default: None)
        
        """
        
        #
        # Generate figure with subplots
        # note that some of the plot windows are not used... maybe later
        #
        app = pg.mkQApp()
        win = pg.GraphicsWindow()  #'%s  %s (%s)' % (cell.status['name'], cell.status['modelType'], cell.status['species']))
        self.win = win
        win.resize(1000, 800)
        Vplot = win.addPlot(labels={'left': 'Vm (mV)', 'bottom': 'Time (ms)'})
        rightGrid = win.addLayout(rowspan=2)
        win.nextRow()
        PostCellPlot = win.addPlot(labels={'left': 'Ipost (nA)', 'bottom': 'Time (ms)'})
        win.nextRow()
        Iplot = win.addPlot(labels={'left': 'Iinj (nA)', 'bottom': 'Time (ms)'})
        
        # right side:
        IVplot = rightGrid.addPlot(labels={'left': 'Vm (mV)', 'bottom': 'Icmd (nA)'})
        IVplot.showGrid(x=True, y=True)
        rightGrid.nextRow()
        spikePlot = rightGrid.addPlot(labels={'left': 'Iinj (nA)', 'bottom': 'Spike times (ms)'})
        rightGrid.nextRow()
        FIplot = rightGrid.addPlot(labels={'left': 'Spike count', 'bottom': 'Iinj (nA)'})
        
        win.ci.layout.setRowStretchFactor(0, 10)
        win.ci.layout.setRowStretchFactor(1, 5)

        #
        # Plot the simulation results
        #
        Vm = self.sgc_voltage_traces
        Iinj = self.current_traces
        # Icmd = self.current_cmd
        DVm = self.axon_voltage_traces
        post = self.post_cell_current_traces
        calyxca = self.calyx_ca
        t = self.time_values
        steps = len(Iinj)
        # plot I, V traces
        colors = [(i, steps*3./2.) for i in range(steps)]
        
        # plot all the stimuli
        for i in range(steps):
            Vplot.plot(t, Vm[i], pen='w')
            #Iplot.plot(t, Iinj[i], pen=colors[i])  # that was the current pulses
            Iplot.plot(t[:len(calyxca[i])], calyxca[i], pen='y') # calcium is more interesting
            PostCellPlot.plot(t, post[i], pen=colors[i])
            if len(DVm) == 0:
                continue
            nnodes = len(DVm[i])
            axcolors = [pg.intColor(k, nnodes) for k in range(nnodes)]
            for j in range(len(DVm[i])):  # for each site
                if j in range(nnodes): #[1, nnodes-1]:
                    ilen = len(DVm[i][j])
                    Vplot.plot(t[:ilen], DVm[i][j], pen=axcolors[j])
            trange = [0,140]
            xmin = trange[0]
            xmax = trange[1]
            Vplot.setXRange(xmin, xmax)
            print("rescaled voltage")
            Iplot.setXRange(xmin, xmax)
            PostCellPlot.setXRange(xmin, xmax)
        Iplot.setXLink(Vplot)
        PostCellPlot.setXLink(Vplot)
        if rmponly:
            print("In this condition")
            return

    def show_mpl(self, cell=None, rmponly=False):
        
        figure = mpl.figure(constrained_layout=True) # creates box
        figure.patch.set_facecolor((1, 1, 1)) # background color
        ncols = 1
        nrows = 3
        widths = [1]
        heights = [5, 1, 2]
        specs = figure.add_gridspec(ncols=ncols, nrows=nrows, width_ratios=widths,
                                  height_ratios=heights)
        axes = []
        labels = ['Vax (mV)', 'IPost (nV)', r'$Ca{2+}$'] 
        for row in range(nrows):
            for col in range(ncols):
                if row == 0: # adds subplot
                    ax = figure.add_subplot(specs[row, col])
                    axes_row0 = ax
                else:
                    ax = figure.add_subplot(specs[row, col], sharex=axes_row0)
                    
                label = 'Width: {}\nHeight: {}'.format(widths[col], heights[row])
                ax.annotate(labels[row], (1.05, 0.9), xycoords='axes fraction', ha='left', fontsize=7, color='k') # gives labels
                axes.append(ax)
                ax.set_facecolor((1, 1, 1))
                ax.spines['right'].set_visible(False) # removes right axis
                ax.spines['top'].set_visible(False) # removes top axis
                # PH.nice_plot(ax)
            
        Vplot = axes[0]
        PostCellPlot = axes[1]
        Ca_plot = axes[2]
        PostcellPlot = axes[1]
        print("rmpvalue : ")
        print(rmponly)
        """
        Plot results from run_iv()
        Using pyqtgraph.
        
        Parameters
        ----------
        cell : cell object (default: None)
        
        """
        
        # Plot the simulation results
        #
        Vm = self.sgc_voltage_traces
        Iinj = self.current_traces
        DVm = self.axon_voltage_traces
        post = self.post_cell_current_traces
        calyxca = self.calyx_ca
        t = self.time_values
        steps = len(Iinj)
        # plot I, V traces
        color_index = np.linspace(0, 1, steps)
        line_w = 0.4
        for i in range(steps):
            print('i: ', i)
            step_color = mpl.cm.cool(color_index[i])
            step_color2 = mpl.cm.RdGy(color_index[i])
            Vplot.plot(t, Vm[i], color=mpl.cm.cool(i), linewidth=line_w)
            Ca_plot.plot(t[:len(calyxca[i])], calyxca[i], color=step_color, linewidth=line_w) # calcium is more interesting
            PostCellPlot.plot(t, post[i], color=step_color, linewidth=line_w)
            if len(DVm) == 0:
                continue
            nnodes = len(DVm[i])
            axcolor_index = np.linspace(0, 1, nnodes)
        
            for j in range(len(DVm[i])):  # for each site
                if j in range(nnodes): #[1, nnodes-1]:
                    ilen = len(DVm[i][j])
                    Vplot.plot(t[:ilen], DVm[i][j], color=step_color2, linewidth=line_w)
            trange = [0,140]
            xmin = trange[0]
            xmax = trange[1]
            Vplot.set_xlim(xmin, xmax)
            print("rescaled voltage")
        mpl.show()
        if rmponly:
            print("In this condition")
            return
            
        
if __name__ == '__main__':
    M = Model()  # create a model cell
    M.run()
    plot_mode = 'mpl'
    if plot_mode == 'mpl':
        M.show_mpl()
    elif plot_mode == 'pg':
        M.show_pg()
        if sys.flags.interactive == 0:
            pg.QtGui.QApplication.exec_() 
