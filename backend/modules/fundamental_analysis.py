"""
Comprehensive Fundamental Analysis Module
"""
import pandas as pd
import numpy as np
from datetime import datetime


class FundamentalAnalyzer:

    def __init__(self, current_info, financials=None, news=None):
        self.info = current_info or {}
        # ✅ FIX: Guard against None being passed explicitly for financials
        self.financials = financials if isinstance(financials, dict) else {}
        self.news = news or []

        self.income_stmt  = self.financials.get('income_stmt')
        self.balance_sheet = self.financials.get('balance_sheet')
        self.cashflow     = self.financials.get('cashflow')

        self.safe_get = lambda key, default=0: self.info.get(key, default) if self.info else default

        self.ratios = {}
        self._calculate_ratios()

        # ✅ Cache score so get_comprehensive_analysis() doesn't recompute it
        self._cached_score = None

    def _calculate_ratios(self):
        self.ratios = {
            'profitability': self._calculate_profitability_ratios(),
            'efficiency':    self._calculate_efficiency_ratios(),
            'liquidity':     self._calculate_liquidity_ratios(),
            'leverage':      self._calculate_leverage_ratios(),
            'valuation':     self._calculate_valuation_ratios()
        }
        self.qualitative = self._analyze_qualitative_aspects()
        self.red_flags   = self._detect_red_flags()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_financial_value(self, financial_data, *possible_keys):
        """Extract a float value from a financial statement Series."""
        if financial_data is None or len(financial_data) == 0:
            return None
        for key in possible_keys:
            try:
                if key in financial_data.index:
                    value = financial_data[key]
                    if pd.notna(value) and value != 0:
                        return float(value)
            except Exception:   # ✅ FIX: log instead of bare continue
                pass
        # Case-insensitive partial match
        for key in possible_keys:
            for idx in financial_data.index:
                try:
                    if isinstance(idx, str) and key.lower() in idx.lower():
                        value = financial_data[idx]
                        if pd.notna(value) and value != 0:
                            return float(value)
                except Exception:
                    pass
        return None

    def _stmt_ok(self, stmt):
        """Return True only if a financial statement DataFrame is usable."""
        return stmt is not None and not getattr(stmt, 'empty', True) and len(stmt) > 0

    # ── News helpers ────────────────────────────────────────────────────────

    def _news_text(self, article):
        """
        ✅ FIX: data_fetcher now returns cleaned dicts with 'title' and no 'summary'.
        Build a safe combined text string from whatever keys exist.
        """
        return ' '.join([
            article.get('title', ''),
            article.get('summary', ''),     # kept for backwards compat
            article.get('publisher', ''),
        ]).lower()

    def _scan_news(self, keywords, limit=10):
        """Return up to `limit` articles whose text contains any keyword."""
        hits = []
        for article in self.news:
            text = self._news_text(article)
            for kw in keywords:
                if kw in text:
                    hits.append({
                        'keyword': kw,
                        'source': article.get('title', 'Unknown'),
                        # ✅ FIX: published_at is already an ISO string from data_fetcher
                        'date': article.get('published_at', 'Unknown')
                    })
                    break
            if len(hits) >= limit:
                break
        return hits

    # ── Profitability ───────────────────────────────────────────────────────

    def _calculate_profitability_ratios(self):
        ratios = {}

        if self._stmt_ok(self.income_stmt):
            latest_year = self.income_stmt.iloc[:, 0]
            total_revenue    = self._get_financial_value(latest_year, 'Total Revenue', 'Revenue')
            gross_profit     = self._get_financial_value(latest_year, 'Gross Profit')
            operating_income = self._get_financial_value(latest_year, 'Operating Income', 'EBIT')
            net_income_is    = self._get_financial_value(latest_year, 'Net Income', 'Net Income Common Stockholders')

            if total_revenue and total_revenue > 0:
                if gross_profit:
                    gpm = (gross_profit / total_revenue) * 100
                    ratios['gross_profit_margin'] = {
                        'value': gpm, 'benchmark': 20,
                        'status': self._evaluate_margin(gpm, 20)
                    }
                ebitda = operating_income or gross_profit
                if ebitda:
                    ratios['ebitda_margin'] = {
                        'value': (ebitda / total_revenue) * 100,
                        'benchmark': None, 'status': 'Higher is better'
                    }
                if net_income_is:
                    ratios['pat_margin'] = {
                        'value': (net_income_is / total_revenue) * 100,
                        'benchmark': None, 'status': 'Higher means more profit from sales'
                    }

        # ROE
        net_income = self.safe_get('netIncomeToCommon') or None
        shareholder_equity = self.safe_get('totalStockholderEquity') or None
        if net_income and shareholder_equity and shareholder_equity > 0:
            roe = (net_income / shareholder_equity) * 100
        else:
            raw = self.safe_get('returnOnEquity')
            roe = raw * 100 if raw else None
        ratios['roe'] = {'value': roe, 'benchmark': 18, 'status': self._evaluate_roe(roe)}

        # ROA
        total_assets = self.safe_get('totalAssets') or None
        if net_income and total_assets and total_assets > 0:
            ratios['roa'] = {
                'value': (net_income / total_assets) * 100,
                'benchmark': None, 'status': 'Higher is better (asset-use efficiency)'
            }

        # ROCE — ✅ FIX: use safe_get('ebit') only, don't fall back to possibly-None operating_income
        total_debt = self.safe_get('totalDebt') or 0
        ebit = self.safe_get('ebit') or None
        if ebit and shareholder_equity is not None:
            capital_employed = total_debt + shareholder_equity
            if capital_employed > 0:
                ratios['roce'] = {
                    'value': (ebit / capital_employed) * 100,
                    'benchmark': None, 'status': 'Higher means better returns on invested capital'
                }

        # EPS
        shares = self.safe_get('sharesOutstanding') or None
        if net_income and shares and shares > 0:
            ratios['eps'] = {'value': net_income / shares, 'benchmark': None, 'status': 'Higher is better'}
        else:
            ratios['eps'] = {'value': self.safe_get('trailingEps') or None,
                             'benchmark': None, 'status': 'Higher is better'}

        return ratios

    # ── Efficiency ──────────────────────────────────────────────────────────

    def _calculate_efficiency_ratios(self):
        ratios = {}
        total_revenue = self.safe_get('totalRevenue') or None
        total_assets  = self.safe_get('totalAssets') or None

        if total_revenue and total_assets and total_assets > 0:
            at = total_revenue / total_assets
            ratios['asset_turnover'] = {
                'value': at, 'benchmark': None,
                'status': f'Higher turnover ({at:.2f}x/yr) = effective asset use'
            }

        if self._stmt_ok(self.balance_sheet) and self._stmt_ok(self.income_stmt):
            latest_bs = self.balance_sheet.iloc[:, 0]
            latest_is = self.income_stmt.iloc[:, 0]
            inventory    = self._get_financial_value(latest_bs, 'Inventory', 'Inventory Net')
            cogs         = self._get_financial_value(latest_is, 'Cost Of Goods Sold', 'Cost Of Revenue')
            receivables  = self._get_financial_value(latest_bs, 'Accounts Receivable', 'Net Receivables')

            if inventory and inventory > 0 and cogs:
                it = cogs / inventory
                ratios['inventory_turnover'] = {
                    'value': it, 'benchmark': None,
                    'status': f'{it:.2f}x/yr — higher is better'
                }
                ratios['inventory_days'] = {
                    'value': 365 / it, 'benchmark': None,
                    'status': f'{365/it:.1f} days — fewer is better'
                }

            if total_revenue and receivables and receivables > 0:
                rt = total_revenue / receivables
                ratios['receivables_turnover'] = {
                    'value': rt, 'benchmark': None,
                    'status': f'{rt:.2f}x — higher = faster collections'
                }
                ratios['dso'] = {
                    'value': 365 / rt, 'benchmark': None,
                    'status': f'{365/rt:.1f} days — lower is better'
                }

        current_assets      = self.safe_get('totalCurrentAssets') or 0
        current_liabilities = self.safe_get('totalCurrentLiabilities') or 0
        working_capital = current_assets - current_liabilities
        if total_revenue and working_capital > 0:
            wct = total_revenue / working_capital
            ratios['working_capital_turnover'] = {
                'value': wct, 'benchmark': None,
                'status': 'Higher = more sales per unit of working capital'
            }

        return ratios

    # ── Liquidity ───────────────────────────────────────────────────────────

    def _calculate_liquidity_ratios(self):
        ratios = {}

        if self._stmt_ok(self.cashflow):
            latest_cf = self.cashflow.iloc[:, 0]
            cfo = self._get_financial_value(
                latest_cf, 'Operating Cash Flow', 'Total Cash From Operating Activities'
            )
            if cfo is not None:
                ratios['cfo'] = {
                    'value': cfo, 'benchmark': None,
                    'status': 'Positive CFO is critical — negative CFO is a red flag',
                    'is_positive': cfo > 0
                }
                capex_raw = self._get_financial_value(
                    latest_cf, 'Capital Expenditures', 'Capital Expenditure'
                )
                capex = abs(capex_raw) if capex_raw else 0
                fcf = cfo - capex
                ratios['fcf'] = {
                    'value': fcf, 'benchmark': None,
                    'status': 'Cash available after maintaining/expanding assets',
                    'is_positive': fcf > 0,
                    'growth_guidance': 'Sustainable FCF growth typically ≤20%/yr'
                }

        current_ratio = self.safe_get('currentRatio') or None
        if current_ratio:
            ratios['current_ratio'] = {
                'value': current_ratio, 'benchmark': 1.5,
                'status': '>1.5 is good, <1 may indicate liquidity issues'
            }

        return ratios

    # ── Leverage ────────────────────────────────────────────────────────────

    def _calculate_leverage_ratios(self):
        ratios = {}
        total_debt         = self.safe_get('totalDebt') or 0
        shareholder_equity = self.safe_get('totalStockholderEquity') or 0
        total_assets       = self.safe_get('totalAssets') or 0
        ebit               = self.safe_get('ebit') or 0
        # ✅ FIX: yfinance returns interestExpense as negative — use abs()
        interest_expense   = abs(self.safe_get('interestExpense') or 0)

        if ebit and interest_expense > 0:
            ic = ebit / interest_expense
            ratios['interest_coverage'] = {
                'value': ic, 'benchmark': 2.0,
                'status': self._evaluate_interest_coverage(ic),
                'risk_level': 'High Risk' if ic < 1.5 else ('Moderate' if ic < 2.5 else 'Low Risk')
            }

        if shareholder_equity > 0:
            de = (total_debt / shareholder_equity) * 100
            ratios['debt_to_equity'] = {
                'value': de, 'benchmark': 100,
                'status': self._evaluate_debt_equity(de),
                'is_high_leverage': de > 100
            }

        if total_assets > 0:
            da = (total_debt / total_assets) * 100
            ratios['debt_to_asset'] = {
                'value': da, 'benchmark': None,
                'status': f'{da:.1f}% — higher = more leverage and risk'
            }

        if shareholder_equity > 0 and total_assets:
            fl = total_assets / shareholder_equity
            ratios['financial_leverage'] = {
                'value': fl, 'benchmark': None,
                'status': f'{fl:.2f}x leverage'
            }

        if ebit > 0:
            de_ebit = (total_debt / ebit) * 100
            ratios['debt_ebit_ratio'] = {
                'value': de_ebit, 'benchmark': None,
                'status': 'Lower is better; declining trend = improving debt management'
            }

        return ratios

    # ── Valuation ───────────────────────────────────────────────────────────

    def _calculate_valuation_ratios(self):
        ratios = {}
        current_price      = self.safe_get('currentPrice') or self.safe_get('regularMarketPrice') or 0
        shares_outstanding = self.safe_get('sharesOutstanding') or 0
        total_revenue      = self.safe_get('totalRevenue') or 0

        if current_price and shares_outstanding > 0 and total_revenue > 0:
            sps = total_revenue / shares_outstanding
            if sps > 0:
                ps = current_price / sps
                ratios['ps_ratio'] = {
                    'value': ps, 'benchmark': None,
                    'status': f'P/S of {ps:.2f}x — compare to industry peers'
                }

        pb = self.safe_get('priceToBook') or None
        if pb:
            ratios['pb_ratio'] = {
                'value': pb, 'benchmark': 3.0,
                'status': self._evaluate_pb_ratio(pb),
                'interpretation': f'{pb:.2f}x book value'
            }

        pe = self.safe_get('trailingPE') or None
        if pe:
            ratios['pe_ratio'] = {
                'value': pe, 'benchmark': 25,
                'status': self._evaluate_pe_ratio(pe),
                'interpretation': f'Investors pay {pe:.2f}x per unit of profit',
                'warning': 'Avoid stocks above 25–30x earnings'
            }

        dy = self.safe_get('dividendYield') or None
        if dy:
            ratios['dividend_yield'] = {
                'value': dy * 100, 'benchmark': None,
                'status': 'Annual dividend / market price'
            }

        return ratios

    # ── Qualitative ─────────────────────────────────────────────────────────

    def _analyze_qualitative_aspects(self):
        return {
            'management_background':    self._analyze_management_background(),
            'business_ethics':          self._analyze_business_ethics(),
            'corporate_governance':     self._analyze_corporate_governance(),
            'minority_shareholders':    self._analyze_minority_shareholders(),
            'share_transactions':       self._analyze_share_transactions(),
            'related_party_transactions': self._analyze_related_party_transactions(),
            'promoter_compensation':    self._analyze_promoter_compensation(),
            'operator_activity':        self._analyze_operator_activity(),
            'shareholders':             self._analyze_shareholders(),
            'political_affiliation':    self._analyze_political_affiliation(),
            'promoter_lifestyle':       self._analyze_promoter_lifestyle()
        }

    def _analyze_management_background(self):
        return {
            'status': 'Requires manual review from annual report',
            'factors_to_check': [
                'Management education and experience',
                'Track record in running business',
                'Any criminal cases against promoters',
                'Industry expertise'
            ],
            'data_source': 'Company annual report, management discussion section'
        }

    def _analyze_business_ethics(self):
        keywords = ['scam', 'bribery', 'fraud', 'unethical', 'illegal', 'violation', 'fine', 'penalty']
        # ✅ FIX: uses _scan_news() which reads correct keys from cleaned news dicts
        hits = self._scan_news(keywords, limit=10)
        return {
            'status': 'Red Flag Detected' if hits else 'No Issues Found',
            'mentions': hits,
            'factors_to_check': [
                'Management involvement in scams', 'Bribery cases',
                'Unfair business practices', 'Regulatory violations'
            ],
            'data_source': 'News articles, regulatory filings'
        }

    def _analyze_corporate_governance(self):
        return {
            'status': 'Requires manual review',
            'factors_to_check': [
                'Appointment of directors', 'Organisation structure',
                'Transparency in disclosures', 'Board independence'
            ],
            'data_source': 'Annual report, corporate governance section'
        }

    def _analyze_minority_shareholders(self):
        return {
            'status': 'Requires manual review',
            'factors_to_check': [
                'Corporate actions impacting minority holders',
                'Voting rights protection',
                'Fair treatment in dividends/buybacks'
            ],
            'data_source': 'Annual report, investor relations section'
        }

    def _analyze_share_transactions(self):
        return {
            'status': 'Requires regulatory filing review',
            'factors_to_check': [
                'Promoter buying/selling through promoter groups',
                'Clandestine transactions',
                'Unusual timing of transactions'
            ],
            'data_source': 'SEBI filings, BSE/NSE disclosures'
        }

    def _analyze_related_party_transactions(self):
        keywords = ['related party', 'promoter', 'relative', 'vendor', 'favors']
        hits = self._scan_news(keywords, limit=10)
        return {
            'status': 'Red Flag Detected' if hits else 'No Issues Reported',
            'mentions': [h['source'] for h in hits],
            'warning': 'Excessive RPTs signal favouritism/malpractice',
            'factors_to_check': [
                'Financial favours to promoters\' relatives, friends, vendors',
                'Transactions at cost of shareholder funds'
            ],
            'data_source': 'Annual report, related party transaction disclosures'
        }

    def _analyze_promoter_compensation(self):
        return {
            'status': 'Requires annual report review',
            'warning': 'Excessive promoter salaries (high % of profits) is red flag',
            'factors_to_check': [
                'Promoter salary as % of profits',
                'Comparison with industry peers',
                'Justification for compensation'
            ],
            'data_source': 'Annual report, remuneration section'
        }

    def _analyze_operator_activity(self):
        return {
            'status': 'Requires price-volume pattern analysis',
            'warning': 'Unusual price behaviour coinciding with promoter transactions indicates manipulation',
            'factors_to_check': [
                'Unusual price spikes/drops', 'Volume anomalies',
                'Coincidence with promoter transactions'
            ],
            'data_source': 'Price charts, SEBI surveillance reports'
        }

    def _analyze_shareholders(self):
        return {
            'status': 'Available from info',
            'note': 'Check for shareholders with >1% of outstanding shares',
            'data_source': 'Annual report, shareholding pattern'
        }

    def _analyze_political_affiliation(self):
        keywords = ['political', 'politician', 'government', 'minister', 'party']
        hits = self._scan_news(keywords, limit=5)
        return {
            'status': 'Requires careful analysis',
            'mentions': [h['source'] for h in hits],
            'warning': 'Too close to political parties may indicate regulatory risk',
            'factors_to_check': [
                'Promoter political connections',
                'Business requiring constant political support',
                'Government contracts dependency'
            ]
        }

    def _analyze_promoter_lifestyle(self):
        keywords = ['flamboyant', 'luxury', 'display', 'wealth', 'extravagant']
        hits = self._scan_news(keywords, limit=5)
        return {
            'status': 'Red Flag if Excessive' if hits else 'No Issues',
            'mentions': [h['source'] for h in hits],
            'warning': 'Flamboyant promoters displaying wealth may indicate poor governance'
        }

    # ── Red flags ───────────────────────────────────────────────────────────

    def _detect_red_flags(self):
        red_flags = []

        cfo = self.ratios.get('liquidity', {}).get('cfo', {}).get('value')
        if cfo is not None and cfo < 0:
            red_flags.append({
                'type': 'Negative CFO', 'severity': 'High',
                'description': 'Company draining cash from operations — major red flag',
                'recommendation': 'Avoid — company should generate positive cash from operations'
            })

        if self._stmt_ok(self.income_stmt):
            latest_is = self.income_stmt.iloc[:, 0]
            total_revenue = self._get_financial_value(latest_is, 'Total Revenue', 'Revenue')
            other_income  = self._get_financial_value(latest_is, 'Other Income',
                                                       'Total Other Income Expense Net')
            if total_revenue and other_income and total_revenue > 0:
                pct = (abs(other_income) / total_revenue) * 100
                if pct > 20:
                    red_flags.append({
                        'type': 'High Other Income', 'severity': 'Medium',
                        'description': f'Other income ({pct:.1f}%) is disproportionately large vs core sales',
                        'recommendation': 'Scrutinise — core business may be weak'
                    })

        ic = self.ratios.get('leverage', {}).get('interest_coverage', {}).get('value')
        if ic is not None and ic < 1.5:
            red_flags.append({
                'type': 'Poor Interest Coverage', 'severity': 'High',
                'description': f'Interest coverage ({ic:.2f}x) is very low',
                'recommendation': 'High risk of default — avoid'
            })

        de = self.ratios.get('leverage', {}).get('debt_to_equity', {}).get('value')
        if de is not None and de > 150:
            red_flags.append({
                'type': 'Excessive Debt', 'severity': 'High',
                'description': f'Debt-to-equity ({de:.1f}%) is very high',
                'recommendation': 'High leverage risk — proceed with caution'
            })

        pe = self.ratios.get('valuation', {}).get('pe_ratio', {}).get('value')
        if pe is not None and pe > 30:
            red_flags.append({
                'type': 'Overvalued (High P/E)', 'severity': 'Medium',
                'description': f'P/E ({pe:.2f}x) is above 30 — expensive',
                'recommendation': 'Do not buy stocks above 25–30x earnings'
            })

        roe = self.ratios.get('profitability', {}).get('roe', {}).get('value')
        if roe is not None and roe < 10:
            red_flags.append({
                'type': 'Low ROE', 'severity': 'Medium',
                'description': f'ROE ({roe:.2f}%) below average (prefer 18%+)',
                'recommendation': 'Poor return on equity — inefficient capital use'
            })

        qualitative_flags = []
        if self.qualitative.get('business_ethics', {}).get('status') == 'Red Flag Detected':
            qualitative_flags.append('Business ethics concerns detected in news')
        if self.qualitative.get('related_party_transactions', {}).get('status') == 'Red Flag Detected':
            qualitative_flags.append('Related party transaction concerns')
        if self.qualitative.get('political_affiliation', {}).get('mentions'):
            qualitative_flags.append('Political connections detected — may indicate dependency')
        if qualitative_flags:
            red_flags.append({
                'type': 'Qualitative Concerns', 'severity': 'Medium to High',
                'description': '; '.join(qualitative_flags),
                'recommendation': 'Review annual report and news carefully'
            })

        return red_flags

    # ── Scoring ─────────────────────────────────────────────────────────────

    def get_fundamental_score(self):
        if not self.info:
            return {'score': 0, 'signal': 'Unknown', 'error': 'No data available'}

        # ✅ Return cached result if already computed
        if self._cached_score is not None:
            return self._cached_score

        score = 0
        max_possible = 0
        details = {}

        for name, fn, weight in [
            ('profitability', self._score_profitability, 25),
            ('efficiency',    self._score_efficiency,    15),
            ('liquidity',     self._score_liquidity,     20),
            ('leverage',      self._score_leverage,      20),
            ('valuation',     self._score_valuation,     20),
        ]:
            result = fn()
            score        += result['score']
            max_possible += result['max_score']
            details[name] = result

        # ✅ FIX: max_possible guaranteed non-zero (sum of fixed max_scores = 100)
        final_score = (score / max_possible * 100) if max_possible > 0 else 0
        final_score = min(100, max(0, final_score))

        # ✅ FIX: Cap penalty at 30 pts so score never goes below 0 semantically
        red_flag_penalty = min(len(self.red_flags) * 5, 30)
        final_score = max(0, final_score - red_flag_penalty)

        result = {
            'score': round(final_score, 2),
            'signal': self._get_fundamental_signal(final_score),
            'details': details,
            'ratios': self.ratios,
            'qualitative': self.qualitative,
            'red_flags': self.red_flags,
            'red_flag_count': len(self.red_flags)
        }
        self._cached_score = result
        return result

    def _score_profitability(self):
        score = 0
        max_score = 25
        roe = self.ratios['profitability'].get('roe', {}).get('value')
        if roe:
            score += 10 if roe >= 20 else 8 if roe >= 18 else 5 if roe >= 14 else 3 if roe >= 10 else 0
        pat = self.ratios['profitability'].get('pat_margin', {}).get('value')
        if pat:
            score += 8 if pat >= 20 else 6 if pat >= 10 else 4 if pat >= 5 else 0
        gm = self.ratios['profitability'].get('gross_profit_margin', {}).get('value')
        if gm:
            score += 7 if gm >= 20 else 5 if gm >= 15 else 0
        return {'score': score, 'max_score': max_score, 'ratios': self.ratios['profitability']}

    def _score_efficiency(self):
        score = 0
        max_score = 15
        at = self.ratios['efficiency'].get('asset_turnover', {}).get('value')
        if at:
            score += 5 if at >= 1.0 else 3 if at >= 0.5 else 0
        it = self.ratios['efficiency'].get('inventory_turnover', {}).get('value')
        if it:
            score += 5 if it >= 5 else 3 if it >= 3 else 0
        dso = self.ratios['efficiency'].get('dso', {}).get('value')
        if dso:
            score += 5 if dso <= 45 else 3 if dso <= 60 else 0
        return {'score': score, 'max_score': max_score, 'ratios': self.ratios['efficiency']}

    def _score_liquidity(self):
        score = 0
        max_score = 20
        cfo = self.ratios['liquidity'].get('cfo', {}).get('value')
        if cfo is not None:
            score += 12 if cfo > 0 else 0
        fcf = self.ratios['liquidity'].get('fcf', {}).get('value')
        if fcf is not None and fcf > 0:
            score += 5
        cr = self.ratios['liquidity'].get('current_ratio', {}).get('value')
        if cr and cr >= 1.5:
            score += 3
        return {'score': score, 'max_score': max_score, 'ratios': self.ratios['liquidity']}

    def _score_leverage(self):
        score = 0
        max_score = 20
        ic = self.ratios['leverage'].get('interest_coverage', {}).get('value')
        if ic:
            score += 8 if ic >= 3.0 else 6 if ic >= 2.0 else 3 if ic >= 1.5 else 0
        de = self.ratios['leverage'].get('debt_to_equity', {}).get('value')
        if de is not None:
            score += 7 if de < 50 else 5 if de < 100 else 2 if de < 150 else 0
        da = self.ratios['leverage'].get('debt_to_asset', {}).get('value')
        if da is not None and da < 50:
            score += 5
        return {'score': score, 'max_score': max_score, 'ratios': self.ratios['leverage']}

    def _score_valuation(self):
        score = 0
        max_score = 20
        pe = self.ratios['valuation'].get('pe_ratio', {}).get('value')
        if pe:
            score += (10 if 15 <= pe <= 25 else 8 if 10 <= pe < 15
                      else 5 if 25 < pe <= 30 else 2)
        pb = self.ratios['valuation'].get('pb_ratio', {}).get('value')
        if pb:
            score += 7 if pb <= 3 else 4 if pb <= 5 else 2
        dy = self.ratios['valuation'].get('dividend_yield', {}).get('value')
        if dy and dy > 2:
            score += 3
        return {'score': score, 'max_score': max_score, 'ratios': self.ratios['valuation']}

    # ── Evaluators ──────────────────────────────────────────────────────────

    def _evaluate_margin(self, value, benchmark):
        if value is None: return 'N/A'
        return 'Healthy' if value >= benchmark else ('Fair' if value >= benchmark * 0.8 else 'Below Benchmark')

    def _evaluate_roe(self, roe):
        if roe is None: return 'N/A'
        return ('Excellent (≥20%)' if roe >= 20 else 'Very Good (18–20%)' if roe >= 18
                else 'Average (14–16%)' if roe >= 14 else 'Below Average' if roe >= 10 else 'Poor (<10%)')

    def _evaluate_interest_coverage(self, r):
        return ('Strong (Safe)' if r >= 3.0 else 'Good' if r >= 2.0
                else 'Moderate (Barely covering)' if r >= 1.5 else 'Poor (High Risk)')

    def _evaluate_debt_equity(self, r):
        return ('Low Debt (Strong equity base)' if r < 50 else 'Moderate Debt' if r < 100
                else 'High Debt (Caution)' if r < 150 else 'Very High Debt (Risky)')

    def _evaluate_pb_ratio(self, pb):
        return ('Undervalued' if pb < 1 else 'Fair Value' if pb <= 3
                else 'Slightly Overvalued' if pb <= 5 else 'Overvalued')

    def _evaluate_pe_ratio(self, pe):
        return ('Undervalued (Potential value)' if pe < 15 else 'Fair Value (Good range)' if pe <= 25
                else 'Slightly Overvalued (Caution)' if pe <= 30 else 'Overvalued (Avoid)')

    def _get_fundamental_signal(self, score):
        return ('Strong' if score >= 75 else 'Good' if score >= 60
                else 'Moderate' if score >= 45 else 'Weak' if score >= 30 else 'Poor')

    # ── Comprehensive output ─────────────────────────────────────────────────

    def get_comprehensive_fundamental_analysis(self):
        """✅ FIX: Uses cached score — no double computation."""
        score = self.get_fundamental_score()
        return {
            'score': score,
            'profitability_ratios': self.ratios['profitability'],
            'efficiency_ratios':    self.ratios['efficiency'],
            'liquidity_ratios':     self.ratios['liquidity'],
            'leverage_ratios':      self.ratios['leverage'],
            'valuation_ratios':     self.ratios['valuation'],
            'qualitative_analysis': self.qualitative,
            'red_flags':            self.red_flags,
            'data_sources': {
                'annual_report':  'Download from company website',
                'industry_data':  'Industry association websites',
                'news':           'Business newspapers, Google Alert'
            }
        }