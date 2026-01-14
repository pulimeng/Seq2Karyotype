from collections import defaultdict, namedtuple
import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import scipy.stats as sts
import pandas as pd
import numpy as np
import argparse as agp
import matplotlib.cm as cm
import matplotlib.patches as patches

from S2K import Testing
from S2K import Consts

#colorsCN = {}
colorsCN = defaultdict(lambda: 'purple')
colorsCN['A'] = 'lime'
colorsCN['AA'] = 'blue'
colorsCN['AAB'] = 'cyan'
colorsCN['(AB)(2+n)'] = 'black'
colorsCN['(AB)(2-n)'] = 'orange'
colorsCN['AB'] = 'darkgray'
colorsCN['AAAB'] = 'magenta'
colorsCN['AAA'] = 'brown'
colorsCN['AAAA'] = 'darkolivegreen'
colorsCN['A+AA'] = 'orchid'
colorsCN['AA+AAA'] = 'pink'
colorsCN['AA+AAAA'] = 'gold'
colorsCN['AA+AAB'] = 'teal'
colorsCN['AAB+AAAB'] = 'turquoise'
colorsCN['AAB+AABB'] = 'green'
colorsCN['UN'] = 'skyblue'

def complex_evolution(row):
    if row['model'] == 'A' and row['chrom'] != 'chrY' and row['chrom'] != 'chrX':
        clonality = row['k']
        tmpcn = clonality * 1 + (1 - clonality) * 2
        tmpai = clonality / (4 - 2 * clonality)
        deltacn = abs(tmpcn - row['cn'])/row['cn']
        deltaai = abs(tmpai - row['ai'])/row['ai']
        if deltacn > 0.2 or deltaai > 0.1:
            return True
        else:
            return False
    else:
        return False

def meerkat_plot (bed_df, axs, chrom_sizes, cn_max, model_conf_thr=0.1):
    chrs = chrom_sizes.index.values.tolist()

    chrs.sort (key = Consts.CHROM_ORDER.index)
    start = 0
    axs[1].plot ((start, start), (0, 5), 'k:', lw = 0.5)
    axs[0].plot ((start, start), (0, 1.0), 'k:', lw = 0.5)
    mids = []
    
    for chrom in chrs:
        bed_chr = bed_df.loc[bed_df['chrom'] == chrom]
        for _, b in bed_chr.iterrows():
            if b['cn'] < 0.6:
                color = 'purple'
            elif b['cn'] > 0.6 and b['cn'] <= 4.5:
                color = colorsCN[b['model']]
            elif b['cn'] > 4.5 and b['cn'] < 10:
                color = 'yellow'
            else:
                color = 'red'
            
            alpha = 1
            
            if b['model'] in ['AA+AAB', 'AAB+AAAB', 'AAB+AABB', 'A+AA', 'AA+AAA', 'AA+AAAA', 'AAA+AAAA']:
                ms = b['model'].split('+')
                if b['model'] == 'AAB+AABB':
                    ms = ['AAB', '(AB)(2+n)']
                axs[0].fill_between ((start + b['start'], start + b['end']),
                                     (b['k'], b['k']), color = colorsCN[ms[1]], alpha = alpha)
                axs[0].fill_between ((start + b['start'], start + b['end']),(b['k'], b['k']),
                                     (1, 1), color = colorsCN[ms[0]], alpha = alpha)
            else:
                axs[0].fill_between ((start + b['start'], start + b['end']),
                                     (b['k'], b['k']), color = color, alpha = alpha)
            axs[1].fill_between (x = (start + b['start'], start + b['end']),
                                 y1 = (b['cn'], b['cn']), y2 = (2, 2), color = color, alpha = alpha)
            if b['model_fitness'] < model_conf_thr and b['model'] != 'AB':
                axs[0].scatter((start + (b['start'] + b['end']) / 2), b['k'] + 0.05, marker='v', color='red', edgecolor='black', s=15)
        
        complex_rows = bed_chr[bed_chr.apply(lambda row: complex_evolution(row), axis=1) & ((1-bed_chr['cent'])*bed_chr['size'] >= 1e6)]
        if len(complex_rows) > 0:
            for _, b in complex_rows.iterrows():
                bar_center = start + (b['start'] + b['end']) / 2
                bar_height = b['k']  
                bar_width = b['end'] - b['start']
                rect = patches.Rectangle((bar_center - bar_width / 2, 0), bar_width, bar_height, linewidth=1, edgecolor='red', facecolor='none')
                patch = axs[0].add_patch(rect)
                patch.set_zorder(20)  # Set the z-order here

        end = chrom_sizes[chrom]
        mids.append (start + end / 2)
        start += end
        axs[1].plot ((start, start), (0.0, 5.0), 'k:', lw = 0.5)
        axs[0].plot ((start, start), (0.0, 1.0), 'k:', lw = 0.5)        
        

    axs[-1].set_xticks (mids)
    axs[-1].set_xticklabels (chrs, rotation = 60)
        
    axs[1].plot ((0, start), (2, 2), 'k--', lw = 1)        
        
    axs[0].set_ylim ((-0.009,  1.05))
    axs[0].set_xlim ((-3e7, start + 3e7))
    
    default_ylim = (1, 3)
    
    cn_min = bed_df.cn.min()
    
    lower_limit = min(cn_min * 0.8, default_ylim[0]*0.8)
    if cn_max < 4.5:
        upper_limit = max(cn_max * 1.05, default_ylim[1]*1.05)
    else:
        upper_limit = 5*1.05
    
    axs[1].set_ylim((lower_limit, upper_limit))
        
    axs[0].set_ylabel ('clonality')
    axs[1].set_ylabel ('copy number')

def reporting_plot (bed_df, axs, chrom_sizes):
    chrs = chrom_sizes.index.values.tolist()
    chrs.sort(key = Consts.CHROM_ORDER.index)
    start = 0
    axs[1].plot ((start, start), (0, 4), 'k:', lw = 0.5)
    axs[0].plot ((start, start), (0, 0.95), 'k:', lw = 0.5)
    mids = []
    
    for chrom in chrs:
        for _, b in bed_df.loc[bed_df['chrom'] == chrom].iterrows():
            a = 1
            color = colorsCN[b['model']]
            k = b['k']
            
            if b['model'] in ['AA+AAB', 'AAB+AAAB', 'AAB+AABB', 'A+AA', 'AA+AAA']:
                ms = b['model'].split('+')
                if b['model'] == 'AAB+AABB':
                    ms = ['AAB', '(AB)(2+n)']
                axs[0].fill_between ((start + b['start'], start + b['end']),
                                     (k, k), color = colorsCN[ms[1]], alpha = a)
                axs[0].fill_between ((start + b['start'], start + b['end']),(k, k),
                                     (1, 1), color = colorsCN[ms[0]], alpha = a)
            else:
                axs[0].fill_between ((start + b['start'], start + b['end']),
                                         (k, k), color = color, alpha = a)
            # axs[0].fill_between ((start + b['start'], start + b['end']), (k, k), color = color, alpha = a)
            axs[1].fill_between (x = (start + b['start'], start + b['end']), y1 = (b['cn'], b['cn']), y2 = (2, 2), color = color, alpha = a)
           
        end = chrom_sizes[chrom]
        mids.append (start + end / 2)
        start += end
        
        axs[1].plot ((start, start), (0.0, 4), 'k:', lw = 0.5)
        axs[0].plot ((start, start), (0.0, 0.95), 'k:', lw = 0.5)        
    
    axs[-1].set_xticks (mids)
    axs[-1].set_xticklabels (chrs, rotation = 60)
        
    axs[1].plot ((0, start), (2, 2), 'k--', lw = 1)        
    
    ranges = bed_df.loc[~(bed_df['k'].isnull()), ['k','m']].agg ([min, max])
    maxk = max (bed_df.loc[~(bed_df['k'].isnull()), 'k'].max(), -bed_df.loc[~(bed_df['k'].isnull()), 'k'].min())
    

    #axs[0].set_ylim ((-0.009, 1.01))
    axs[0].set_ylim ((-0.009,  maxk *1.1))
    axs[0].set_xlim ((-3e7, start + 3e7))
    axs[1].set_ylim (bed_df.cn.agg([min,max]).values*np.array((0.9,1.1)))
        
    axs[0].set_ylabel ('clonality')
    axs[1].set_ylabel ('copy number')

def leopard_plot (bed_df, params, ax, highlight = '', color_norm = 'black', color_hit = 'darkred', alpha = 1):
    
    a,b,bt = params
    
    x = np.log10 (bed_df['size'])
    y = np.log10 (np.abs (bed_df['k']))
    ax.plot (x, y, marker = 'o', c = color_norm, lw = 0, alpha = alpha)

    x = np.log10 (bed_df.loc[(bed_df.status != 'norm'), 'size'])
    y = np.log10 (np.abs(bed_df.loc[(bed_df.status != 'norm'), 'k']))
    ax.plot (x, y, marker = 'o', c = color_hit, lw = 0, alpha = alpha)
    

    for chrom in highlight:
        tmp = bed_df.loc[bed_df['chrom'] == chrom].sort_values (by = 'start')
        x = np.log10 (tmp['size'])
        y = np.log10 (np.abs(tmp['k']))
        ax.plot (x, y, marker = 's', c = 'magenta', lw = 1, alpha = 1, fillstyle = 'none')


    xt = np.linspace (-3, 2.5, 10)
    ax.plot (xt, -a*xt - b, c = color_norm)
    ax.plot (xt, -a*xt - bt , c = color_hit, linestyle = ':')

    ax.set_xlabel ('size (MB) / log')
    ax.set_ylabel ('clonality / log')

def plot_cdf (all_values, ax,  all_colors, par = (1,1), n = 100, xscale = 'lin', half = False):
    #l = 0.6*(max(values) - min(values))
    #x = np.linspace ((max(values) + min(values))/2 - l, (max(values) + min(values))/2 + l, n)
    xmax = par[0] + 3*par[1]
    
    if half:
        xmin = par[0]# - 3*par[1]
        x = np.linspace (xmin, xmax, n)
        y = 2*sts.norm.cdf (x, par[0], par[1]) - 1
    else:
        xmin = par[0] - 3*par[1]
        x = np.linspace (xmin, xmax, n)
        y = sts.norm.cdf (x, par[0], par[1])
    
    values = all_values[(all_values >= xmin)&(all_values <= xmax)]
    colors = all_colors[(all_values >= xmin)&(all_values <= xmax)]
    
    if xscale == 'lin':
        ax.scatter (np.sort(values), np.linspace (0.01,0.99, len(values)),
                    c = [colors[i] for i in np.argsort(values)])
        ax.plot (x, y, 'r-')
    elif xscale == 'log':
        ax.scatter (np.log10(np.sort(values)), np.linspace (0.01,0.99, len(values)),
                    c = [colors[i] for i in np.argsort(values)])
        ax.plot (np.log10(x), y, 'r-')
    else:
        raise ('Unknown scale')
        

def earth_worm_plot (data_df, bed_df, params, chrom, axs, markersize = 2):

    chromdata = data_df.loc[data_df.chrom == chrom]

    chromdata.loc[chromdata['symbol'] == 'E'].plot(x = 'position', y = 'vaf', lw = 0, alpha = 0.3,
                                                   color = 'orange', marker = '.', 
                                                   ms = markersize, ax = axs[0], legend = False)
    chromdata.loc[chromdata['symbol'] == 'U'].plot(x = 'position', y = 'vaf', lw = 0, alpha = 0.3,
                                                   color = 'darkgray', marker = '.',
                                                   ms = markersize, ax = axs[0], legend = False)
    chromdata.loc[chromdata['symbol'] == 'N'].plot(x = 'position', y = 'vaf', lw = 0, alpha = 0.3,
                                                   color = 'blue', marker = '.',
                                                   ms = markersize, ax = axs[0], legend = False)


    chromdata.loc[chromdata['symbol'] == 'N'].plot(x = 'position', y = 'cov', lw = 0, alpha = 0.3,
                                                   color = 'red', marker = '.',
                                                   ms = markersize, ax = axs[1], legend = False)
    chromdata.loc[chromdata['symbol'] == 'U'].plot(x = 'position', y = 'cov', lw = 0, alpha = 0.3,
                                                   color = 'darkorange', marker = '.',
                                                   ms = markersize, ax = axs[1], legend = False)
    chromdata.loc[chromdata['symbol'] == 'E'].plot(x = 'position', y = 'cov', lw = 0, alpha = 0.3,
                                                   color = 'darkgray', marker = '.',
                                                   ms = markersize, ax = axs[1], legend = False)
    
    axs[1].plot ((0, chromdata.position.max()), (params['m0'], params['m0']), 'k--', lw=1)
    
    if chromdata['cov'].max() < 5 * params['m0']:
        axs[1].set_ylim (chromdata['cov'].min(), chromdata['cov'].max())
    else:
        axs[1].set_ylim (chromdata['cov'].min(), 5*params['m0'])
    
    axs[0].set_ylabel ('BAF')
    axs[1].set_ylabel ('cov.')
    
    chrombed = bed_df.loc[(bed_df.chrom == chrom)]
    
    if len(chrombed) > 0:
        default_ylim = (1, 3)
        
        cn_min = chrombed.cn.min()
        cn_max = chrombed.cn.max()
        
        lower_limit = min(cn_min * 0.8, default_ylim[0]*0.8)
        if cn_max < 4.5:
            upper_limit = max(cn_max * 1.05, default_ylim[1]*1.05)
        else:
            upper_limit = 6*1.05
        
        axs[3].set_ylim((lower_limit, upper_limit))
            
    for _, seg in chrombed.loc[(chrombed.cent < 0.5) & (chrombed['size'] >= 0.95)].iterrows():
        plot_cn = seg.cn if seg.cn < 5 else 5
        linestyle = '-' if seg['size'] > 5 else ':'
        
        if seg['cn'] > 0.6 and seg['cn'] <= 4.5:
            axs[2].plot((seg.start, seg.end), (seg.k, seg.k), c=colorsCN[seg.model], lw=1, ls=linestyle, marker='o', markersize=3)
            axs[3].plot((seg.start, seg.end), (plot_cn, plot_cn), c=colorsCN[seg.model], lw=1, ls=linestyle)
        elif seg['cn'] < 0.6:
            axs[2].plot ((seg.start, seg.end), (seg.k, seg.k), c='purple', lw=1, marker='o', ls=linestyle, markersize=3)
            axs[3].plot ((seg.start, seg.end), (seg.cn, seg.cn), c='purple', ls=linestyle, lw=1)
            text_x = (seg.start + seg.end) / 2  # Middle of the segment
            text_y = plot_cn + 0.2  # Offset to position text above the line
            axs[3].text(text_x, text_y, str(round(seg.cn,1)), color='purple', ha='center', fontsize=6)
            # axs[3].text(text_x, text_y, 'Del²', color='purple', ha='center', fontsize=6)
        elif seg['cn'] <= 10 and seg['cn'] > 4.5:
            axs[2].plot ((seg.start, seg.end), (seg.k, seg.k), c=colorsCN[seg.model], lw=1, ls=linestyle, marker='o', markersize=3)
            axs[3].plot ((seg.start, seg.end), (5, 5), c='yellow', lw=1, ls=linestyle)
            axs[2].plot ((seg.start, seg.end), (seg.k, seg.k), c='yellow', lw = 5, alpha = .6)
            text_x = (seg.start + seg.end) / 2  # Middle of the segment
            text_y = 5 - 0.7  # Offset to position text above the line
            axs[3].text(text_x, text_y, str(round(seg.cn)), color='black', ha='center', fontsize=6)
        else:
            axs[2].plot ((seg.start, seg.end), (seg.k, seg.k), c='red', lw=1, ls=linestyle, marker='o', markersize=3)
            axs[3].plot ((seg.start, seg.end), (5, 5), c='red', lw=1, ls=linestyle)
            text_x = (seg.start + seg.end) / 2  # Middle of the segment
            text_y = 5 + 0.2  # Offset to position text above the line
            axs[3].text(text_x, text_y, str(round(seg.cn)), color='red', ha='center', fontsize=6)
        
        if seg['model'] != 'AB':
            bar_position = (seg.start + seg.end) / 2  # Position at the center of the segment
            confidence_level = seg.model_fitness  # Confidence level for the segment
            cmap = cm.gray_r  # Grayscale colormap (reversed)
            color = cmap(confidence_level)
            bar_width = seg.end - seg.start  # Width of the bar equals the size of the segment
            axs[4].bar(bar_position, 1, color=color, width=bar_width, linewidth=0)
        
        if complex_evolution(seg):
            bar_width = seg['end'] - seg['start']
            rect1 = patches.Rectangle((seg['start'], seg['k'] - 0.1), bar_width, 0.2, linewidth=1, edgecolor='red', facecolor='none')
            patch1 = axs[2].add_patch(rect1)
            rect2 = patches.Rectangle((seg['start'], seg['cn'] - upper_limit/10), bar_width, upper_limit/5, linewidth=1, edgecolor='red', facecolor='none')
            patch2 = axs[3].add_patch(rect2)
            patch1.set_zorder(20)  # Set the z-order here
            patch2.set_zorder(20)  # Set the z-order here

    axs[2].set_ylim ((-0.05,  1.05))
    axs[3].plot ((0, chromdata.position.max()), (2, 2), 'k--', lw=1)
    axs[4].set_ylim ((0.0,  1.0))

    axs[2].set_ylabel ('clonality')
    axs[3].set_ylabel ('cn')
    axs[4].set_ylabel('fitness')

def check_solution_plot_opt (bed, ax, model_thr,
                             highlight = [], xcol = 'cn'):
    
    
    for _, b in bed.loc[bed['model'].notna(),:].iterrows():

        ec = 'w' if b['model_dipscore'] < model_thr else 'orange'
        if b['chrom'] == 'chrX':
            ax.scatter (b[xcol],b['ai'], c = colorsCN[b['model']], s = b['size']*2,
                        edgecolor = ec, marker = 'X')
        elif b['chrom'] == 'chrY':
            ax.scatter (b[xcol],b['ai'], c = colorsCN[b['model']], s = b['size']*4,
                        edgecolor = ec, marker = 'v')
        else:
                        ax.scatter (b[xcol],b['ai'], c = colorsCN[b['model']],
                                    s = b['size'], 
                                    edgecolor = ec,
                                    marker = 'o')

    highlight_filter = [c in highlight for c in bed.chrom.tolist()]
    x = bed.loc[(highlight_filter), xcol].values
    y = bed.loc[(highlight_filter), 'ai'].values
    
    ax.plot (x, y, marker = 's', c = 'darkorange', lw = 0, alpha = 1, fillstyle = 'none')
    
    ax.set_xlabel ('Coverage/copy number')
    ax.set_ylabel ('Allelic imbalance')


def verification_plot_CNV (d_ch, ch_bed, ax, par, type = 'CDF', column = 'vaf', no_bins = 100):
    assert type in ["CDF", "PDF"], "Unknown plot type!"
    assert column in ['vaf', 'cov'], "Unknown column!"
    ##Plot reference
    
    vaf_down_lim, vaf_up_lim = sts.norm.ppf ((0.0005,0.9995), par['v0'], np.sqrt(0.25/par['m0']*par['fb']))
     
    if column == 'vaf':
        x = np.linspace (0,1, 500)
        ref_norm = sts.norm.cdf (x, par['v0'], np.sqrt(0.25/par['m0']*par['fb']))
        xlim = (-0.01,1.01)
    elif column == 'cov':
        x = Testing.lambda_ppf (np.linspace (0,1,500), par['m0'], par['l'])
        ref_norm = np.linspace (0,1,500)
        xlim = (d_ch['cov'].quantile ([0.005, 0.999]))
        
    if type == "CDF":
        ax.plot (x, ref_norm,
                 color = 'red', lw = 0.5)
        ax.set_xlim (xlim)
    
    ax.plot ((),(), lw = 2, color = 'red', label = 'diploid reference')
    
    ##Plot AB
    dipl_bed = ch_bed.loc[ch_bed['model'] == 'AB']
    starts = dipl_bed['start'].values
    ends = dipl_bed['end'].values
    pos_filt = ((d_ch.position.values[:, np.newaxis] > starts[np.newaxis,:]) &\
                (d_ch.position.values[:, np.newaxis] < ends[np.newaxis,:])).any (axis = 1) 
    
    tmp = d_ch.loc[(d_ch['symbol'] == Consts.E_SYMBOL)&(pos_filt)]
        
    v,c = np.unique (tmp[column].values, return_counts = True)
    if type == "CDF":
        ax.plot(v, np.cumsum(c)/np.sum(c), '.', markersize = 0.5, color = colorsCN['AB'])
        ax.plot ((),(), lw = 2, label = 'AB', color = colorsCN['AB'])
    else:
        ax.hist (v, bins = np.linspace (0,1, no_bins), lw = 2, 
                 histtype = "step", density = True, color = colorsCN['AB'])
        ax.plot ((),(), lw = 2, label = 'AB', color = colorsCN['AB'])
    
    ##Plot CNVs
    CNV_bed = ch_bed.loc[ch_bed['model'] != 'AB']
    for _, cb in CNV_bed.iterrows():
        #(d_ch['symbol'] == cb['symbol'])&\
        
        tmp = d_ch.loc[(d_ch['vaf'] < vaf_up_lim)&(d_ch['vaf'] > vaf_down_lim)&\
                       (d_ch['position'] >= cb['start'])&\
                       (d_ch['position'] <= cb['end'])]
        v,c = np.unique (tmp[column].values, return_counts = True)
        if type == "CDF":
            ax.plot(v, np.cumsum(c)/np.sum(c), '.', markersize = 1,
                    color = colorsCN[cb['model']])
            ax.plot ((),(), lw = 2, color = colorsCN[cb['model']],
                     label = cb['chrom']+':'+str(cb['start'])+'-'+str(cb['end'])+':'+cb['model'])
        else:
            ax.hist (v, bins = np.linspace (0,1, no_bins), lw = 2, 
                     histtype = "step", density = True,
                     color = colorsCN[cb['model']])
            ax.plot ((),(), lw = 2, color = colorsCN[cb['model']],
                     label = cb['chrom']+':'+str(cb['start'])+'-'+str(cb['end'])+':'+cb['model'])
    
    ax.legend()