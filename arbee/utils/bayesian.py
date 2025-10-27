"""
Bayesian Mathematics Utilities
Implements log-likelihood ratio aggregation with correlation adjustments
"""
import math
from typing import List, Dict, Any, Tuple
import numpy as np
from config.settings import settings
from config.system_constants import (
    PROB_CLAMP_MIN,
    PROB_CLAMP_MAX,
    POSTERIOR_CLAMP_MIN,
    POSTERIOR_CLAMP_MAX,
    MIN_RECENCY_WEIGHT,
    SENSITIVITY_LLR_MULTIPLIERS,
    SENSITIVITY_WEAKEST_REMOVAL_PCT,
    DEFAULT_VERIFIABILITY_SCORE,
    DEFAULT_INDEPENDENCE_SCORE,
    DEFAULT_RECENCY_SCORE
)


class BayesianCalculator:
    """
    Bayesian probability calculator using log-likelihood ratios

    Formulas:
    - log_odds = ln(p / (1-p))
    - posterior_log_odds = prior_log_odds + Σ(adjusted_LLR)
    - p_post = exp(log_odds_post) / (1 + exp(log_odds_post))
    - adjusted_LLR = LLR × verifiability × independence × recency
    """

    @staticmethod
    def prob_to_log_odds(p: float) -> float:
        """
        Convert probability to log-odds

        Args:
            p: Probability (0 < p < 1)

        Returns:
            Log-odds value
        """
        # Clamp to avoid division by zero
        p = max(min(p, PROB_CLAMP_MAX), PROB_CLAMP_MIN)
        return math.log(p / (1 - p))

    @staticmethod
    def log_odds_to_prob(log_odds: float) -> float:
        """
        Convert log-odds back to probability

        Args:
            log_odds: Log-odds value

        Returns:
            Probability (0-1)
        """
        try:
            return math.exp(log_odds) / (1 + math.exp(log_odds))
        except OverflowError:
            # Handle extreme values
            return 1.0 if log_odds > 0 else 0.0

    @staticmethod
    def adjust_llr(
        llr: float,
        verifiability: float,
        independence: float,
        recency: float
    ) -> float:
        """
        Adjust LLR based on quality scores

        Args:
            llr: Raw log-likelihood ratio
            verifiability: Verifiability score (0-1)
            independence: Independence score (0-1)
            recency: Recency score (0-1)

        Returns:
            Adjusted LLR

        Note:
            Recency is treated as a bonus, not a penalty. Even older evidence
            gets at least MIN_RECENCY_WEIGHT (60%) to avoid over-discounting
            valuable evidence. This prevents the system from under-reacting to
            genuine strong evidence just because it's not brand new.
        """
        # Make recency a bonus, not a strict penalty
        # Even old evidence gets minimum weight
        recency_adjusted = max(recency, MIN_RECENCY_WEIGHT)
        weight = verifiability * independence * recency_adjusted
        return llr * weight

    @staticmethod
    def apply_correlation_shrinkage(
        llrs: List[float],
        cluster_size: int
    ) -> List[float]:
        """
        Apply shrinkage to correlated evidence

        Args:
            llrs: List of LLRs in correlated cluster
            cluster_size: Number of correlated items

        Returns:
            Shrunk LLRs
        """
        if cluster_size <= 1:
            return llrs

        shrinkage_factor = 1.0 / math.sqrt(cluster_size)
        return [llr * shrinkage_factor for llr in llrs]

    @classmethod
    def aggregate_evidence(
        cls,
        prior_p: float,
        evidence_items: List[Dict[str, Any]],
        correlation_clusters: List[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate evidence using Bayesian updating

        Args:
            prior_p: Prior probability (0-1)
            evidence_items: List of dicts with keys:
                {LLR, verifiability, independence, recency, id}
            correlation_clusters: Optional list of correlated evidence (IDs or indices)
                Can be List[List[str]] (evidence IDs) or List[List[int]] (indices)

        Returns:
            Dict with posterior probability and details
        """
        # Convert prior to log-odds
        log_odds_prior = cls.prob_to_log_odds(prior_p)

        # Process each evidence item
        adjusted_llrs = []
        evidence_summary = []

        # Build ID -> index mapping for cluster resolution
        id_to_index = {}
        for i, item in enumerate(evidence_items):
            item_id = item.get('id', f'evidence_{i}')
            id_to_index[item_id] = i

        for i, item in enumerate(evidence_items):
            llr = item.get('LLR', 0.0)
            verif = item.get('verifiability_score', item.get('verifiability', DEFAULT_VERIFIABILITY_SCORE))
            indep = item.get('independence_score', item.get('independence', DEFAULT_INDEPENDENCE_SCORE))
            recency = item.get('recency_score', item.get('recency', DEFAULT_RECENCY_SCORE))

            # Adjust LLR by quality scores
            weight = verif * indep * recency
            adjusted_llr = cls.adjust_llr(llr, verif, indep, recency)

            adjusted_llrs.append(adjusted_llr)
            evidence_summary.append({
                'id': item.get('id', f'evidence_{i}'),
                'LLR': llr,
                'weight': weight,
                'adjusted_LLR': adjusted_llr
            })

        # Apply correlation adjustments
        if correlation_clusters:
            for cluster in correlation_clusters:
                # Convert cluster to indices (handle both string IDs and integer indices)
                cluster_indices = []
                for item in cluster:
                    if isinstance(item, int):
                        # Already an index
                        if 0 <= item < len(evidence_items):
                            cluster_indices.append(item)
                    elif isinstance(item, str):
                        # Evidence ID - look up index
                        if item in id_to_index:
                            cluster_indices.append(id_to_index[item])
                    # Skip invalid items

                if not cluster_indices:
                    continue  # Skip empty clusters

                cluster_llrs = [adjusted_llrs[i] for i in cluster_indices]
                shrunk_llrs = cls.apply_correlation_shrinkage(
                    cluster_llrs,
                    len(cluster_indices)
                )

                # Update adjusted LLRs
                for idx, shrunk_llr in zip(cluster_indices, shrunk_llrs):
                    adjusted_llrs[idx] = shrunk_llr
                    evidence_summary[idx]['adjusted_LLR'] = shrunk_llr

        # Calculate posterior
        total_llr = sum(adjusted_llrs)
        log_odds_posterior = log_odds_prior + total_llr
        p_bayesian = cls.log_odds_to_prob(log_odds_posterior)

        # Clamp extreme probabilities
        p_bayesian = max(min(p_bayesian, POSTERIOR_CLAMP_MAX), POSTERIOR_CLAMP_MIN)

        return {
            'p0': prior_p,
            'log_odds_prior': log_odds_prior,
            'evidence_summary': evidence_summary,
            'total_adjusted_LLR': total_llr,
            'log_odds_posterior': log_odds_posterior,
            'p_bayesian': p_bayesian
        }

    @classmethod
    def sensitivity_analysis(
        cls,
        prior_p: float,
        evidence_items: List[Dict[str, Any]],
        scenarios: List[Tuple[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run sensitivity analysis on different assumptions

        Args:
            prior_p: Prior probability
            evidence_items: Evidence list
            scenarios: List of (name, multiplier) tuples for LLR adjustments

        Returns:
            List of scenario results
        """
        if scenarios is None:
            scenarios = list(SENSITIVITY_LLR_MULTIPLIERS) + [
                (f'remove weakest {int(SENSITIVITY_WEAKEST_REMOVAL_PCT * 100)}%', None)
            ]

        results = []

        for scenario_name, multiplier in scenarios:
            if 'remove weakest' in scenario_name.lower():
                # Remove weakest evidence by weight
                sorted_items = sorted(
                    evidence_items,
                    key=lambda x: x.get('verifiability_score', DEFAULT_VERIFIABILITY_SCORE) *
                                  x.get('independence_score', DEFAULT_INDEPENDENCE_SCORE) *
                                  x.get('recency_score', DEFAULT_RECENCY_SCORE),
                    reverse=True
                )
                keep_fraction = 1.0 - SENSITIVITY_WEAKEST_REMOVAL_PCT
                cutoff = int(len(sorted_items) * keep_fraction)
                modified_items = sorted_items[:cutoff]
            else:
                # Multiply all LLRs
                modified_items = [
                    {**item, 'LLR': item['LLR'] * multiplier}
                    for item in evidence_items
                ]

            result = cls.aggregate_evidence(prior_p, modified_items)
            results.append({
                'scenario': scenario_name,
                'p': result['p_bayesian']
            })

        return results


class KellyCalculator:
    """
    Kelly Criterion calculator for optimal bet sizing
    """

    @staticmethod
    def kelly_fraction(
        edge: float,
        odds: float = 1.0,
        max_fraction: float = None
    ) -> float:
        """
        Calculate Kelly fraction for bet sizing

        Args:
            edge: Expected edge (p_true - p_market)
            odds: Odds ratio (typically 1.0 for prediction markets)
            max_fraction: Maximum allowed fraction (default 5% conservative Kelly)

        Returns:
            Recommended fraction of bankroll to bet
        """
        if edge <= 0 or odds <= 0:
            return 0.0

        # Use config default if no max_fraction provided
        if max_fraction is None:
            max_fraction = settings.MAX_KELLY_FRACTION

        # Standard Kelly: f = edge / odds
        # For prediction markets with p in [0,1]: f = edge / (1 - p_market)
        kelly = edge / (1.0 - edge + odds) if odds != 1.0 else edge

        # Cap at max_fraction for risk management
        return min(kelly, max_fraction)

    @staticmethod
    def expected_value(
        p_true: float,
        p_market: float,
        transaction_costs: float = 0.0,
        slippage: float = 0.0
    ) -> float:
        """
        Calculate expected value per dollar bet

        Args:
            p_true: Estimated true probability
            p_market: Market-implied probability
            transaction_costs: Fees as fraction (e.g., 0.02 for 2%)
            slippage: Estimated slippage as fraction

        Returns:
            Expected value per dollar
        """
        edge = p_true - p_market
        costs = transaction_costs + slippage

        return edge - costs

    @classmethod
    def calculate_stake(
        cls,
        bankroll: float,
        p_true: float,
        p_market: float,
        transaction_costs: float = 0.0,
        slippage: float = 0.0,
        max_kelly: float = None
    ) -> Dict[str, float]:
        """
        Calculate recommended stake size

        Args:
            bankroll: Total available bankroll
            p_true: Estimated true probability
            p_market: Market price
            transaction_costs: Fee fraction
            slippage: Slippage fraction
            max_kelly: Max Kelly fraction

        Returns:
            Dict with edge, kelly_fraction, stake, and expected_value
        """
        edge = p_true - p_market
        ev = cls.expected_value(p_true, p_market, transaction_costs, slippage)

        # Use config default if no max_kelly provided
        if max_kelly is None:
            max_kelly = settings.MAX_KELLY_FRACTION

        if ev <= 0:
            return {
                'edge': edge,
                'expected_value_per_dollar': ev,
                'kelly_fraction': 0.0,
                'suggested_stake': 0.0
            }

        kelly = cls.kelly_fraction(edge, max_fraction=max_kelly)
        stake = bankroll * kelly

        return {
            'edge': edge,
            'expected_value_per_dollar': ev,
            'kelly_fraction': kelly,
            'suggested_stake': stake
        }
