import sys
import numpy as np
import pandas as pd
import scipy.stats as sts
import scipy.optimize as opt
from collections import namedtuple

from S2K import Testing
from S2K import Distribution
from S2K import Segment
from S2K import Run
from S2K import Consts
from S2K import Report

Run_treshold =  namedtuple('Run_treshold', [Consts.N_SYMBOL, Consts.E_SYMBOL])


class Chromosome:
    """Class to contain data, and find runs."""
    def __init__ (self, name, data, config, logger, genome_medians, CB, models):
        self.name = name
        self.data = data
        self.config = config
        self.logger = logger.getChild(f'{self.__class__.__name__}-{self.name}')
        self.genome_medians = genome_medians
        self.CB = CB
        self.models = models
        if name != 'chrY':
            self.cent = (CB.loc[(CB['gieStain'] == 'acen') | (CB['gieStain'] == 'gvar'),'chromStart'].min(),
                     CB.loc[(CB['gieStain'] == 'acen') | (CB['gieStain'] == 'gvar'),'chromEnd'].max())
        else:
            self.cent = (CB.loc[(CB['gieStain'] == 'acen') ,'chromStart'].min(),
                         CB.loc[(CB['gieStain'] == 'acen'),'chromEnd'].max())
         
        self.logger.debug (f"Object chromosome {name} created.")
        
    def markE_onHE(self, he_parameters, z_thr = Consts.HE_Z_THR):
        self.logger.debug (f"Marking {self.name} based on HE test.")
        zv = (self.data['vaf'].values - he_parameters['vaf']) / np.sqrt (0.25/(he_parameters['cov']))
        zc = (self.data['cov'].values - he_parameters['cov']) / np.sqrt (he_parameters['b']*he_parameters['cov'])
        z = zv**2+zc**2
 
        self.data['symbol'] = [Consts.N_SYMBOL if zi >= z_thr else Consts.E_SYMBOL for zi in z]
        self.logger.info (f"""Chromosome {self.name} marked based on parameters
                            v = {he_parameters['vaf']}, c = {he_parameters['cov']}.
                            #N = {sum(self.data['symbol'] == Consts.N_SYMBOL)}, #E = {sum(self.data['symbol'] == Consts.E_SYMBOL)}""")
        
    def mark_on_full_model (self, m):
        self.logger.debug (f'Marking {self.name} based on full model')
        self.get_fragments (n = int(self.config['Segmenting']['No_SNPs']))
        self.get_vaf_shift_full ()
        
        if self.dv_dist is not None:
            indexes, merged_string = Run.merge_symbols (self.dv_dist.string, outliers_threshold = 4)
            for r in indexes:
                start = self.windows_positions[r[0]][0]
                end = self.windows_positions [r[1]][1]
                try:
                    chi2, vaf, fb = Testing.VAF_test (self.data.loc[(self.data['position'] >= start)&(self.data['position'] <= end),],
                                                    m, run_fb = False)
                except:
                    chi2 = 0.0
                    

                ai = np.median (self.dv[r[0]:r[1]])
                outlier = (ai > 0.07) & ((chi2 > float(self.config['VAF']['chi2_high'])) | (chi2 == 0))
                        
                if outlier:
                    self.logger.info (f'Region {self.name}:{start}-{end}, chi2 = {chi2}, marked as U.')
                    self.data.loc[(self.data['position'] >= start)&\
                                        (self.data['position'] <= end), 'symbol'] = Consts.U_SYMBOL
                    
            self.logger.info (f"""{self.name} composition: 
                            #N = {sum(self.data.symbol == Consts.N_SYMBOL)},
                            #E = {sum(self.data.symbol == Consts.E_SYMBOL)},
                            #U = {sum(self.data.symbol == Consts.U_SYMBOL)}""")    
        else:
            self.logger.info (f"All marked {Consts.U_SYMBOL} due to assertion error")
            self.data['symbol'] = Consts.U_SYMBOL
            
            
    def get_fragments (self, n = 1000):
        tmp = self.data
        N = np.max((int(np.floor(len(tmp)/n)),1))
        
        indexes = np.linspace (0,len (tmp), 2*N+2, dtype = int)
        self.indexes = indexes
        self.windows = []
        self.windows_positions = []

        for i in np.arange (2*N):
            tmpi = tmp.iloc[indexes[i]:indexes[i+2], ]
            self.windows.append(tmpi)
            self.windows_positions.append ((tmpi['position'].min(), tmpi['position'].max()))
        
    def get_vaf_shift_full (self, z_thr = 2.5):
        
        def vaf_cdf (v, dv, a, lerr, f, vaf, b):
            cnai = vaf_cnai (v, dv, f, vaf, b, cov)
            cnHO = vaf_HO (v, lerr)
            return a*cnHO + (1 - a)*cnai 
        
        v0 = self.genome_medians['HE']['vaf']
        b = self.genome_medians['HE']['b']
        cov = self.genome_medians['HE']['cov']
        dvs = []
        v0s = []        
        
        for window in self.windows:
            vaf = window.loc[~window.vaf.isna(), 'vaf'].values
            v, c = np.unique(vaf, return_counts = True)

            cnor = np.cumsum(c)/np.sum(c)
            ones0 = c[v >= (cov-1)/cov].sum()
            try:
                f0 = c[v < v0].sum()/(c.sum() - ones0) 
            
                dv0 = v0 - np.median (v[v < v0])
              
                popt, pcov = opt.curve_fit (vaf_cdf, v, cnor, p0 = [dv0, ones0/c.sum(), 2, f0, 0.5, b], 
                                            bounds = ((0,   0,   1, 0, 0.45, 1),
                                                      (0.5, 0.95, 5, 1, 0.55, 10)))
                dvs.append (popt[0])
                v0s.append (popt[-1])
            except (RuntimeError, ValueError):
                dvs.append (0)
                
        dva = np.array (dvs)
        median = np.median(dva[dva > 0])
        dva[dva == 0] = median
        self.dv = dva
        self.v0 = np.array(v0s)
        try:
            self.dv_dist = Distribution.Distribution (self.dv,
                                                  p_thr = 0.1, thr_z = z_thr)
        except (AssertionError):
            self.dv_dist = None
            
    def generate_segments (self, window_size):
        """Method to generate genomic segments"""
        self.segments = []
        self.snvs = []
        self.wrong_segments = []
        subwindows = []
        
        current = 0
        end = self.data['position'].max()
        while current < end:
            window_end = min(current + window_size, end)
            subwindows.append((current, window_end))
            current = window_end
            
        for start, end in subwindows:
            data_view = self.data.loc[(self.data['position'] >= start) &\
                                      (self.data['position'] <= end)]
            if len(data_view) == 0:
                self.logger.error(f"No data in segement {start}-{end})")
                self.wrong_segments.append((self.name, start, end))
                continue
            elif len(data_view.loc[data_view['vaf'] < 1, 'symbol'].value_counts()) == 0:
                self.logger.error(f"No SNVs in segement {start}-{end} with VAF smaller than threshold")
                self.wrong_segments.append((self.name, start, end))
                continue
            else:
                centromere_fraction = (min(end, self.cent[1]) - max(self.cent[0],start))/(end - start)
                cytobands = self.CB[(self.CB['chromStart'] < end)&(self.CB['chromEnd'] > start)].sort_values (by = 'chromStart')['name'].values
                if len(cytobands) > 1:
                    cytobands_str = cytobands[0] + '-' + cytobands[-1]
                else:
                    cytobands_str = cytobands[0]
                    
                self.segments.append(Segment.Segment (data = data_view, 
                                                config = self.config, 
                                                logger = self.logger, 
                                                genome_medians = self.genome_medians,
                                                centromere_fraction = 0 if (centromere_fraction < 0) | (centromere_fraction > 1) else centromere_fraction,
                                                cytobands = cytobands_str,
                                                models=self.models))
                self.snvs.append(data_view)
    
    def debug_merges(self, merges):
        """Debug the merge segments to find the problem"""
        self.logger.info(f"Debugging {len(merges)} merge segments for {self.name}")
        
        for i, (start, end) in enumerate(merges):
            if end <= start:
                self.logger.error(f"PROBLEMATIC MERGE #{i}: start={start}, end={end}, length={end-start}")
                
                # Check previous and next segments for context
                if i > 0:
                    prev_start, prev_end = merges[i-1]
                    self.logger.error(f"  Previous segment: {prev_start}-{prev_end}")
                if i < len(merges) - 1:
                    next_start, next_end = merges[i+1]
                    self.logger.error(f"  Next segment: {next_start}-{next_end}")
        
        # Also check for any other issues
        zero_length = sum(1 for start, end in merges if end == start)
        negative_length = sum(1 for start, end in merges if end < start)
        valid_length = sum(1 for start, end in merges if end > start)
        
        self.logger.info(f"Merge segments summary: {valid_length} valid, {zero_length} zero-length, {negative_length} negative-length")
    
    def generate_merged_segments(self, merges):
        """Method to generate genomic segments with robust error handling"""
        self.merged_segments = []
        self.merged_snvs = []
        self.merged_wrong_segments = []
        
        for i, (start, end) in enumerate(merges):
            try:
                # Check for invalid segment coordinates
                if end <= start:
                    self.logger.error(f"Invalid segment coordinates: start={start}, end={end} (end <= start)")
                    self.merged_wrong_segments.append((self.name, start, end, "Invalid coordinates"))
                    continue
                    
                # Check segment length
                segment_length = end - start
                if segment_length == 0:
                    self.logger.error(f"Zero-length segment: {self.name}:{start}-{end}")
                    self.merged_wrong_segments.append((self.name, start, end, "Zero length"))
                    continue
                    
                data_view = self.data.loc[(self.data['position'] >= start) & 
                                         (self.data['position'] <= end)]
                
                if len(data_view) == 0:
                    self.logger.error(f"No data in segment {self.name}:{start}-{end}")
                    self.merged_wrong_segments.append((self.name, start, end, "No data"))
                    continue
                    
                # Check for valid SNVs
                valid_snvs = data_view.loc[data_view['vaf'] < 1]
                if len(valid_snvs) == 0:
                    self.logger.error(f"No valid SNVs in segment {self.name}:{start}-{end}")
                    self.merged_wrong_segments.append((self.name, start, end, "No valid SNVs"))
                    continue
                    
                # Calculate centromere fraction safely
                cent_overlap_start = max(self.cent[0], start)
                cent_overlap_end = min(end, self.cent[1])
                cent_overlap_length = max(0, cent_overlap_end - cent_overlap_start)
                
                centromere_fraction = cent_overlap_length / segment_length
                
                # Get cytobands
                cytobands = self.CB[(self.CB['chromStart'] < end) & 
                                   (self.CB['chromEnd'] > start)].sort_values(by='chromStart')['name'].values
                
                if len(cytobands) > 1:
                    cytobands_str = cytobands[0] + '-' + cytobands[-1]
                elif len(cytobands) == 1:
                    cytobands_str = cytobands[0]
                else:
                    cytobands_str = "unknown"
                    self.logger.warning(f"No cytoband information for segment {self.name}:{start}-{end}")
                    
                # Create segment
                segment = Segment.Segment(
                    data=data_view, 
                    config=self.config, 
                    logger=self.logger, 
                    genome_medians=self.genome_medians,
                    centromere_fraction=centromere_fraction,
                    cytobands=cytobands_str,
                    models=self.models
                )
                
                self.merged_segments.append(segment)
                self.merged_snvs.append(data_view)
                
                self.logger.debug(f"Successfully created segment {i}: {self.name}:{start}-{end}")
                
            except Exception as e:
                self.logger.error(f"Unexpected error processing segment {self.name}:{start}-{end}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.merged_wrong_segments.append((self.name, start, end, f"Error: {str(e)}"))
    
    
    def report (self, report_type = 'bed'):
        return Report.Report(report_type).chromosome_report(self.merged_segments)

def vaf_cnai (v, dv, a, vaf,b, cov):
    s = np.sqrt((vaf - dv)*(vaf + dv)/(b*cov))
    return a*sts.norm.cdf (v, vaf - dv, s) + (1-a)*sts.norm.cdf (v, vaf + dv, s)

def vaf_HO (v, lerr):
    err = 10**lerr
    return np.exp ((v-1)*err)

def lin (x,a,b):
    return a*x+b
