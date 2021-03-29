from abc import ABC, abstractmethod
from typing import NamedTuple
from functools import reduce
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy_financial as npfa


class PropertyInfo(NamedTuple):
    acq_px: float
    acq_px_allin: float
    acq_dt: pd.Timestamp
    annual_px_inc: float
    pxs: pd.Series
    tx_cost_factor: float

class CustomerCashFlowParams(NamedTuple):
    payback_period: int
    net_rental_yld: float
    initial_cust_share: float
    annual_rent_inc: float
    annual_rpmt_inc: float

class OwnershipFlows(NamedTuple):
    rent: pd.Series
    repayment: pd.Series
    cust_equity_share: pd.Series
    cust_investment: pd.Series
    cmpy_investment: pd.Series


class CashInFlows(NamedTuple):
    rent: pd.Series
    repayment: pd.Series

class OwnershipData(NamedTuple):
    cust_equity_share: pd.Series
    cust_investment: pd.Series
    cmpy_investment: pd.Series

class AcquisitionType(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def undisturbed_model(self, pr_inf: PropertyInfo, cf_params: CustomerCashFlowParams):
        pass
    @abstractmethod
    def realized_model(self):
        pass



def month_end_date_range(start, years_offset):
    initial_month_end = start
    start_month = start
    if start.is_month_end:
        start_month += pd.tseries.offsets.MonthEnd(1)
    else:
        start_month += pd.tseries.offsets.MonthEnd(2)
        initial_month_end = start + pd.tseries.offsets.MonthEnd(1)
    # start_month = start + pd.DateOffset(months=1) # the last date of the following month
    end_month = start_month + pd.DateOffset(years=years_offset)
    dt_range = pd.date_range(start=start_month, end=end_month, freq='M')
    dt_range = dt_range.insert(0, initial_month_end)
    return dt_range


def generate_market_backbone(start_date,drop_pct):
    # random jump process for property market drops & crashes
    drop_years = 20 # 1 drop in 10 years
    drop_prob = 1/(12*drop_years)
    drop_magnitude=drop_pct
    dt_range = month_end_date_range(start_date, 100)
    selloff = np.random.choice(a=[1.0, 1-drop_magnitude], size=dt_range.shape[0], p=[1 - drop_prob, drop_prob])
    market_backbone = pd.Series(
        data=np.cumprod(selloff),
        index=dt_range
    )
    return market_backbone

def generate_property_info(
    min_px, max_px, min_dt, max_dt, annual_px_inc, tx_cost_factor,
    market_backbone = None
):
    acq_px = st.uniform(loc=min_px, scale=max_px-min_px).rvs()
    acq_dt_rng = pd.date_range(min_dt, max_dt, freq='D',closed='left')
    acq_dt = pd.Timestamp(np.random.choice(acq_dt_rng))
    acq_px_allin = acq_px * (1.0 + tx_cost_factor)
    px_dt_range = month_end_date_range(acq_dt, 100)
    pxs_data = np.full(shape=(len(px_dt_range)), fill_value=acq_px)
    rdelta_from_start = [relativedelta(dt, px_dt_range[0]) for dt in px_dt_range]
    months_from_start = np.array([
        max(0, rd.years * 12 + rd.months) for rd in rdelta_from_start
    ])
    monthly_px_inc = (1+annual_px_inc)**(1/12) - 1
    pxs_growth_factors = (1+monthly_px_inc)**months_from_start
    pxs_data = np.multiply(
        pxs_data,
        pxs_growth_factors
    )
    pxs = pd.Series(
        data=pxs_data,
        index=px_dt_range
    )
    if not market_backbone is None:
        pxs = pxs.multiply(market_backbone.reindex(pxs.index))

    return PropertyInfo(
        acq_dt=acq_dt,
        acq_px=acq_px,
        acq_px_allin=acq_px_allin,
        annual_px_inc=annual_px_inc,
        pxs=pxs,
        tx_cost_factor=tx_cost_factor
    )

def sample_property_params(market_backbone):
    min_px=180000.0
    max_px=500000.0
    min_dt=pd.Timestamp("2021-1-1")
    max_dt=pd.Timestamp("2022-1-1")
    annual_px_inc=0.0175
    tx_cost_factor=0.1425
    ppty=generate_property_info(min_px,max_px,min_dt,max_dt,annual_px_inc,tx_cost_factor,market_backbone)
    cfp = CustomerCashFlowParams(
        annual_rent_inc=0.02,
        annual_rpmt_inc=0.04,
        net_rental_yld=0.038,
        payback_period=30,
        initial_cust_share=0.03
    )
    return ppty, cfp

def sample_event(cust_equity_share):
    events = ["none", "buy", "default", "walk_away"]
    p_default = 0.1 # adverse selection ... .1
    p_buy = 0.6
    p_walkaway = 0.3
    p_none = 1.0 - (p_default + p_buy + p_walkaway)
    event_ps = [p_none, p_buy, p_default, p_walkaway]
    urs = st.uniform.rvs()
    event_type = np.random.choice(a=events, size=1, replace=False, p=event_ps)[0]
    start_date = cust_equity_share.index[0]
    if event_type == "none":
        event_date = cust_equity_share.index[-1]
    elif event_type == "buy":
        # buy only if you have more than 10% equity or less than 85%
        buy_idx = (cust_equity_share > .1) & (cust_equity_share < .19)
        buy_idx = cust_equity_share.index[buy_idx]
        event_date = pd.Timestamp(np.random.choice(buy_idx))
    elif event_type == "default":
        # default only if you own less than 85%
        default_idx = (cust_equity_share < .19)
        default_idx = cust_equity_share.index[default_idx][1:] # no default on acq date..
        event_date = pd.Timestamp(np.random.choice(default_idx))
    elif event_type =="walk_away":
        # don't walk away after you own 90%, can not walk away before 5 years
        walk_away_min_date = start_date + pd.offsets.DateOffset(years=5)
        wa_idx = (cust_equity_share < .19) & (cust_equity_share.index > walk_away_min_date)
        wa_idx = cust_equity_share.index[wa_idx]
        event_date = pd.Timestamp(np.random.choice(wa_idx))

    return event_type, event_date

def calc_event_cash_flows(event_type, event_date, owner_flows, property_info):
    event_cf = 0.0
    start_date = owner_flows.cust_equity_share.index[0]
    years_to_event = relativedelta(event_date, start_date).years
    if event_type == 'buy':
        event_cf = owner_flows.cmpy_investment.loc[event_date]
        return event_cf
    elif event_type == 'default':
        if years_to_event < 5:
            event_cf = property_info.pxs.loc[event_date]
            return event_cf
        else:
            event_cf = property_info.pxs.loc[event_date] * (1.0 - owner_flows.cust_equity_share.loc[event_date])
            return event_cf

    elif event_type == 'walk_away':
        wa_fee = 0.05
        wa_fee *= property_info.acq_px_allin
        event_cf = property_info.pxs.loc[event_date] - owner_flows.cust_investment.loc[event_date] + wa_fee
        return event_cf
    elif event_type == "none":
        event_cf = 0.0
        return event_cf
    else:
        raise Exception("event_type not recognized")


def calc_realized_cash_flow(owner_flows, property_info, event_date, event_type):
    first_rent_date = owner_flows.rent.index[0]
    if property_info.acq_dt.is_month_end:
        assert property_info.acq_dt == first_rent_date
    else:
        acq_dt_monthend = property_info.acq_dt + pd.tseries.offsets.MonthEnd(1)
        assert acq_dt_monthend == first_rent_date
    cf_index = owner_flows.rent.index
    cf_data = owner_flows.rent.values + owner_flows.repayment.values
    cf_data[0] -= property_info.acq_px_allin
    cf = pd.Series(
        data = cf_data,
        index = cf_index
    )
    if cf.index.has_duplicates:
        msg = "cf has duplicate"
        raise Exception(msg)

    event_cf = calc_event_cash_flows(event_type, event_date, owner_flows, property_info)

    cf.loc[event_date] += event_cf
    cf = cf.loc[:event_date]
    return cf


def calc_ownership_from_repayment(repayment:pd.Series, pr_inf:PropertyInfo):

    cust_investment = np.cumsum(repayment)
    cmpy_investment = pr_inf.acq_px_allin - cust_investment
    cust_equity_share = cust_investment / pr_inf.acq_px_allin  # this means fees and px paid pack over same time. seems wrong

    # rent = acq_px * net_rental_yld adjusted by annual rent increase
    cust_investment = pd.Series(
        data=cust_investment,
        index=repayment.index,
    )
    cmpy_investment = pd.Series(
        data=cmpy_investment,
        index=repayment.index
    )
    cust_equity_share = pd.Series(
        data=cust_equity_share,
        index=repayment.index
    )
    return OwnershipData(
        cust_investment=cust_investment,
        cmpy_investment=cmpy_investment,
        cust_equity_share=cust_equity_share
    )

def event_repayment_adjustment(event_type, event_date, undisturbed_of:OwnershipFlows):
    rpmt = undisturbed_of.repayment.loc[:event_date].copy()
    rpmt_adjust = np.full(shape=rpmt.shape, fill_value=0.0)
    if event_type == "default":
        n_default_months = 6
        # in the last six months before default event date reduce rent and repayment to zero
        last_rpmt_vals = rpmt.values[-n_default_months:]
        rpmt_adjust[-n_default_months:] = -last_rpmt_vals
    else:
        # no adjustment to inflows necessary
        pass
    rpmt_adjust = pd.Series(
            data=rpmt_adjust,
            index=rpmt.index
        )
    return rpmt_adjust

def event_rent_adjustment(event_type, event_date, undisturbed_of:OwnershipFlows):
    rent = undisturbed_of.rent.loc[:event_date].copy()
    rent_adjust = np.full(shape=rent.shape, fill_value=0.0)
    if event_type == "default":
        n_default_months = 6
        # in the last six months before default event date reduce rent and repayment to zero
        last_rent_vals = rent.values[-n_default_months:]
        rent_adjust[-n_default_months:] = -last_rent_vals
    else:
        # no adjustment to inflows necessary
        pass
    rent_adjust = pd.Series(
            data=rent_adjust,
            index=rent.index
        )
    return rent_adjust

def calc_monthly_rent(pr_inf:PropertyInfo, cf_params:CustomerCashFlowParams, owner_data:OwnershipData):
    cf_dt_range = owner_data.cust_equity_share.index
    year_from_start = np.array([
        max(0, relativedelta(dt, cf_dt_range[1]).years) for dt in cf_dt_range
    ])
    monthly_rent = np.full(
        shape=(len(cf_dt_range)),
        fill_value=pr_inf.acq_px_allin * cf_params.net_rental_yld / 12.0
    )
    monthly_rent = np.array([r * (1 + cf_params.annual_rent_inc) ** y for r, y in zip(monthly_rent, year_from_start)])
    monthly_rent[0] = 0.0  # no rent due on acqn date
    monthly_rent[1:] = np.multiply(monthly_rent[1:], (1.0 - owner_data.cust_equity_share[:-1]))
    monthly_rent = pd.Series(
        data=monthly_rent,
        index=cf_dt_range,
    )  # this leads to "jagged" monthly rent profile, bc after 12months there is a jump due to rent increase
    return monthly_rent

class SimpleEquity(AcquisitionType):
    def __init__(
        self,
        property_info: PropertyInfo,
        cf_params: CustomerCashFlowParams,
    ):
        self.cf_params = cf_params
        self.property_info = property_info
        undisturbed_owner_flows = self.undisturbed_model(property_info, cf_params)
        # sample_event -> buy the property, default, keep going until end
        event_type, event_date = sample_event(undisturbed_owner_flows.cust_equity_share)
        self.event_date = event_date
        self.event_type = event_type
        # determine actual realization of cash flows
        realized_owner_flows, realized_cf = self.realized_model(
            undisturbed_owner_flows,
            event_type,
            event_date
        )
        self.ownership_flows = realized_owner_flows
        self.cf = realized_cf

        # figure out event -> buy the property, default, do nothing
        super().__init__()
    def undisturbed_model(self, pr_inf: PropertyInfo, cf_params: CustomerCashFlowParams):
        cf_dt_range = month_end_date_range(pr_inf.acq_dt, 100)
        year_from_start = np.array([
            max(0, relativedelta(dt, cf_dt_range[1]).years) for dt in cf_dt_range
        ]) # need to start counting the years from the first rent/ repay date, not the aqn date
        # repayment schedule and cust/ cmpy investment schedules
        repayment = np.full(shape=(len(cf_dt_range)), fill_value=np.nan)
        repayment[0] = pr_inf.acq_px_allin * cf_params.initial_cust_share

        initial_cmpy_investment = pr_inf.acq_px_allin - repayment[0]

        initial_mthly_repayment = initial_cmpy_investment / (
            (((1.0 + cf_params.annual_rpmt_inc)**cf_params.payback_period)-1.0) / cf_params.annual_rpmt_inc
        ) / 12.0


        repayment_factor = np.array([
            1+cf_params.annual_rpmt_inc if yr<cf_params.payback_period else 0.0 for yr in year_from_start
        ])

        repayment_factor_series = pd.Series(
            data=repayment_factor,
            index=cf_dt_range
        )

        rpf0idx = repayment_factor_series>0.0
        last_repayment_date = repayment_factor_series.index[rpf0idx][-1]

        repayment[1:] = initial_mthly_repayment
        repayment = np.multiply(repayment, np.power(repayment_factor,year_from_start))

        cust_investment = np.cumsum(repayment)
        cmpy_investment = pr_inf.acq_px_allin - cust_investment

        cust_equity_share = cust_investment / pr_inf.acq_px_allin # this means fees and px paid pack over same time. seems wrong

        # rent = acq_px * net_rental_yld adjusted by annual rent increase
        monthly_rent = np.full(
            shape=(len(cf_dt_range)),
            fill_value=pr_inf.acq_px_allin * cf_params.net_rental_yld / 12.0
        )
        monthly_rent = np.array([ r*(1+cf_params.annual_rent_inc)**y for r,y in zip(monthly_rent, year_from_start)])
        monthly_rent[0] = 0.0  # no rent due on acqn date
        monthly_rent[1:] = np.multiply(monthly_rent[1:], (1.0-cust_equity_share[:-1]))
        monthly_rent = pd.Series(
            data=monthly_rent,
            index=cf_dt_range,
        ) # this leads to "jagged" monthly rent profile, bc after 12months there is a jump due to rent increase
        # rent_idx = monthly_rent<=0
        rent_idx = monthly_rent > 0
        # first_no_rent_idx = monthly_rent.index[rent_idx][0]
        # last_rent_idx = monthly_rent.index[rent_idx][-1]
        last_rent_idx = last_repayment_date
        monthly_rent = monthly_rent.loc[:last_rent_idx]
        repayment = pd.Series(
            data=repayment,
            index=cf_dt_range
        )
        repayment = repayment.loc[:last_rent_idx]
        cust_investment = pd.Series(
            data=cust_investment,
            index=cf_dt_range,
        )
        cust_investment = cust_investment.loc[:last_rent_idx]
        cmpy_investment = pd.Series(
            data=cmpy_investment,
            index=cf_dt_range
        )
        cmpy_investment = cmpy_investment.loc[:last_rent_idx]
        cust_equity_share = pd.Series(
            data=cust_equity_share,
            index=cf_dt_range
        )
        cust_equity_share = cust_equity_share.loc[:last_rent_idx]
        return OwnershipFlows(
            rent=monthly_rent,
            repayment=repayment,
            cust_investment=cust_investment,
            cmpy_investment=cmpy_investment,
            cust_equity_share=cust_equity_share
        )
    def realized_model(
        self,
        undisturbed_of: OwnershipFlows,
        event_type: str,
        event_date: pd.Timestamp,
    ):
        repayment = undisturbed_of.repayment.loc[:event_date].copy()
        rent = undisturbed_of.rent.loc[:event_date].copy()

        rpmt_adjust = event_repayment_adjustment(event_type, event_date, undisturbed_of)
        realized_repayment = repayment + rpmt_adjust

        adj_owner_data = calc_ownership_from_repayment(realized_repayment, self.property_info)

        theo_adj_rent = calc_monthly_rent(self.property_info, self.cf_params, adj_owner_data)

        disturbed_of = OwnershipFlows(
            rent=theo_adj_rent,
            repayment=realized_repayment,
            cust_investment=adj_owner_data.cust_investment,
            cmpy_investment=adj_owner_data.cmpy_investment,
            cust_equity_share=adj_owner_data.cust_equity_share
        )

        rent_adjust = event_rent_adjustment(event_type, event_date, disturbed_of)
        realized_rent = theo_adj_rent + rent_adjust

        repayment_after_rent_shortfall  = realized_repayment + rent_adjust

        adj_owner_data = calc_ownership_from_repayment(repayment_after_rent_shortfall, self.property_info)

        realized_cf = realized_rent + realized_repayment

        adj_owner_flows = OwnershipFlows(
            rent=realized_rent,
            repayment=realized_repayment,
            cust_investment=adj_owner_data.cust_investment,
            cmpy_investment=adj_owner_data.cmpy_investment,
            cust_equity_share=adj_owner_data.cust_equity_share
        )

        event_cf = calc_event_cash_flows(event_type, event_date, adj_owner_flows, self.property_info)
        realized_cf.loc[event_date] += event_cf
        realized_cf.iloc[0] -= self.property_info.acq_px_allin


        return adj_owner_flows, realized_cf

def main():
    total_investments = 0
    aqn_list = []
    cf_list = []
    market_backbone = generate_market_backbone(start_date=pd.Timestamp("2020-12-15"), drop_pct=.220)
    while total_investments<=100000000:
        ppty, cfp = sample_property_params(market_backbone)
        total_investments += ppty.acq_px_allin
        seq = SimpleEquity(ppty, cfp)
        if seq.cf.index.has_duplicates:
            print(seq.cf.index[seq.cf.index.duplicated()])
        assert seq.cf.index[-1] == seq.event_date
        cf_list += [seq.cf]
        aqn_list += [seq]


    total_cf = reduce(lambda x, y: x.add(y,fill_value=0.0), cf_list)
    return total_cf, cf_list, aqn_list, market_backbone





if __name__ == "__main__":
    main()