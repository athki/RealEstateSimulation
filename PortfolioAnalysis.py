import numpy as np
import pandas as pd
import AcquisitionTypes as at
import matplotlib.pyplot as plt
import numpy_financial as npf

from functools import reduce
from pathlib import Path

class PortfolioAnalysis:
    def __init__(self, fund_assets, acq_list, market_backbone, outpath):
        self.fund_assets = fund_assets
        self.acq_list = acq_list
        outpath.mkdir(parents=True, exist_ok=True)
        self.outpath = outpath
        self.market_backbone = market_backbone
    def visualize(self):
        # rent level
        rents_list = [acq.ownership_flows.rent[acq.cf.index] for acq in self.acq_list] # count only the rent payments before the event date
        rent_cf = reduce(lambda x,y: x.add(y,fill_value=0.0), rents_list) # all the rent flows over time as a pd.Series
        print(f"rent_cf = {rent_cf.sum()}")
        fig, ax = plt.subplots()
        ax.plot(rent_cf)
        fig.suptitle("Monthly Rent Levels")
        plt.savefig(self.outpath / "rent_level.pdf")
        plt.close()
        # rental yield = rent_cf / cmpy_investment(1 month previous)
        fig, ax = plt.subplots()
        cmpy_inv_list = [acq.ownership_flows.cmpy_investment[acq.cf.index] for acq in self.acq_list]
        total_cmpy_investment = reduce(lambda x,y: x.add(y,fill_value=0.0), cmpy_inv_list)
        assert total_cmpy_investment.shape == rent_cf.shape
        rent_shifted_data = rent_cf.values[1:]
        cmpy_inv_data = total_cmpy_investment.values[:-1]
        annl_rent_yld_data = np.divide(rent_shifted_data, cmpy_inv_data) * 12.0
        rent_yld = pd.Series(
            data=annl_rent_yld_data,
            index=total_cmpy_investment.index[:-1]
        )
        ax.plot(rent_yld)
        fig.suptitle("Monthly rental yield")
        plt.savefig(self.outpath / "rent_yield.pdf")
        plt.close()

        # cash flows related to repayments


        excess_pmts = {}
        repayment_cf = []
        realized_cf = []

        for acq in self.acq_list:
            # cf = at.calc_realized_cash_flow(acq.ownership_flows, acq.property_info, acq.event_date, acq.event_type)
            cf = acq.cf.copy()
            realized_cf += [cf.copy()]
            cf -= acq.ownership_flows.rent.loc[cf.index] # realized cf without rent
            excess_payment = max(0, np.sum(cf))
            cf.loc[acq.event_date] -= excess_payment # realized cf without excess repayment
            repayment_cf += [cf]
            if not acq.event_date in excess_pmts.keys():
                excess_pmts[acq.event_date] = excess_payment
            else:
                excess_pmts[acq.event_date] += excess_payment

        excess_pmts = pd.Series(excess_pmts).sort_index()
        print(f"excess_pmts = {excess_pmts.sum()}")




        # excess repayments
        fig, ax = plt.subplots()
        ax.plot(excess_pmts)
        fig.suptitle("Excess Repayments")
        plt.savefig(self.outpath / "repayments.pdf")
        plt.close()

        # fund capital position (=-investments + repayments) -> need to eventually also subtract mgt. fees for HH
        fig, ax = plt.subplots()
        rpmt_cf = reduce(lambda x,y: x.add(y, fill_value=0), repayment_cf)
        print(f"rpmt_cf = {rpmt_cf.sum()}")
        fund_cash = self.fund_assets + np.cumsum(rpmt_cf)
        ax.plot(fund_cash)
        fig.suptitle("Fund Cash")
        plt.savefig(self.outpath / "fund_cash.pdf")
        plt.close()

        # total income stream
        fig, ax = plt.subplots()
        ax.plot(np.cumsum(excess_pmts), label="Excess Rpmt", color="green")
        ax.plot(np.cumsum(rent_cf), label="Rental Income", color="red")
        ax.legend()
        fig.suptitle("Income Streams")
        plt.savefig(self.outpath/"income.pdf")
        plt.close()

        # total realized cfs
        trcf = reduce(lambda x,y: x.add(y, fill_value=0.0), realized_cf)
        print(f"trcf = {trcf.sum()}")
        fig, ax = plt.subplots()
        ax.plot(np.cumsum(trcf) + self.fund_assets)
        fig.suptitle("Portfolio Cashflows")
        plt.savefig(self.outpath/"cf_all.pdf")
        plt.close()

        # event payouts
        # all in one

        print(f"sums = {excess_pmts.sum()  + rent_cf.sum() + rpmt_cf.sum()}")


        # separate the total return of the individual options

        dt_dict = {}
        val_dict = {}
        for a in self.acq_list:
            dt = [a.event_date]
            val = [a.cf.sum() / a.property_info.acq_px_allin]
            if a.event_type in dt_dict.keys():
                dt_dict[a.event_type] += dt
            else:
                dt_dict[a.event_type] = dt
            if a.event_type in val_dict.keys():
                val_dict[a.event_type] += val
            else:
                val_dict[a.event_type] = val

        opt_series_dict = {}

        for k in dt_dict.keys():
            curs = pd.Series(data=val_dict[k], index=dt_dict[k]).sort_index()
            opt_series_dict[k] = curs

        fig, ax = plt.subplots()
        for k in dt_dict.keys():
            ax.plot(opt_series_dict[k], label=f"{k} obsd. {opt_series_dict[k].shape[0]}x")
        ax.legend()
        fig.suptitle("Pct Cash Gain of All In Px")
        plt.savefig(self.outpath / "pct_cash_gain.pdf")
        plt.close()

        # plot the market_backbone
        fig,ax = plt.subplots()
        ax.plot(self.market_backbone)
        fig.suptitle("Market Backbone")
        plt.savefig(self.outpath/"mkt_backbone.pdf")
        plt.close()

        pass



def main():
    outpath = Path("~/drop20/")
    outpath.mkdir(parents="True", exist_ok="True")
    print("create the simulations")
    tcf, cfl, aql = at.main()
    irr = npf.irr(tcf.values)
    irr = (1.0 + irr)**12 - 1.0
    print(f"irr={irr:.4f}")
    print("visualize")
    porta = PortfolioAnalysis(fund_assets=100000000, acq_list=aql, outpath=outpath)
    porta.visualize()


if __name__ == "__main__":
    main()
