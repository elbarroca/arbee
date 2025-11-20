"""
Bayesian Mathematics Utilities
Implements log-likelihood ratio aggregation with correlation adjustments
and calibration tracking for prediction validation.
"""
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from datetime import datetime, date
from pydantic import BaseModel, Field
import json
from pathlib import Path
import logging

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

logger = logging.getLogger(__name__)


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
        correlation_adjustments = {
            'method': 'none',
            'details': 'No correlation detected'
        }
        if correlation_clusters:
            cluster_count = len(correlation_clusters)
            total_clustered_items = sum(len(c) for c in correlation_clusters)
            correlation_adjustments = {
                'method': 'shrinkage',
                'details': f'Applied correlation shrinkage to {cluster_count} clusters ({total_clustered_items} items)',
                'cluster_count': cluster_count,
                'total_clustered_items': total_clustered_items
            }
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
            'p_bayesian': p_bayesian,
            'correlation_adjustments': correlation_adjustments
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


# ============================================================================
# Calibration Tracking Module
# ============================================================================

class PredictionRecord(BaseModel):
    """Single prediction record for calibration tracking"""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    market_question: str = Field(..., description="The market question predicted")
    market_slug: str = Field(..., description="Market identifier")
    market_url: Optional[str] = None
    provider: Optional[str] = Field(None, description="e.g., polymarket, kalshi")

    # Prediction details
    predicted_probability: float = Field(..., ge=0.0, le=1.0, description="Our p_bayesian estimate")
    confidence_interval_low: float = Field(..., ge=0.0, le=1.0)
    confidence_interval_high: float = Field(..., ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.80, ge=0.0, le=1.0)

    # Market context
    market_price_at_prediction: Optional[float] = Field(None, description="Market price when we predicted")
    edge: Optional[float] = Field(None, description="p_bayesian - market_price")

    # Resolution
    resolved: bool = Field(default=False)
    resolution_date: Optional[date] = None
    actual_outcome: Optional[bool] = Field(None, description="True if event happened, False if not, None if unresolved")
    resolution_source: Optional[str] = None

    # Metadata
    workflow_id: Optional[str] = None
    evidence_count: Optional[int] = None
    prior_p0: Optional[float] = None
    notes: Optional[str] = None


class CalibrationMetrics(BaseModel):
    """Calibration metrics computed from a set of predictions"""
    total_predictions: int
    resolved_predictions: int
    unresolved_predictions: int

    # Brier score (lower is better, 0 = perfect)
    brier_score: Optional[float] = Field(None, description="Mean squared error of probabilities")

    # Calibration by buckets
    calibration_buckets: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="For each probability bucket (e.g., '50-60%'), actual positive rate"
    )

    # Edge accuracy
    positive_edge_win_rate: Optional[float] = Field(None, description="% of positive edges that won")
    negative_edge_avoid_rate: Optional[float] = Field(None, description="% of negative edges avoided correctly")

    # Confidence interval coverage
    ci_coverage_rate: Optional[float] = Field(None, description="% of outcomes within confidence intervals")

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CalibrationTracker:
    """
    Manages prediction tracking and calibration validation

    This class provides methods to:
    - Store predictions
    - Update with actual outcomes
    - Compute calibration metrics (Brier score, calibration plots)
    - Generate calibration reports

    By default, stores predictions in a JSON file. Can be extended to use a database.
    """

    def __init__(self, storage_path: str = "predictions.json"):
        """
        Initialize calibration tracker

        Args:
            storage_path: Path to JSON file for storing predictions
        """
        self.storage_path = Path(storage_path)
        self.predictions: List[PredictionRecord] = []
        self._load_predictions()

    def _load_predictions(self):
        """Load existing predictions from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.predictions = [PredictionRecord(**p) for p in data]
                logger.info(f"Loaded {len(self.predictions)} predictions from {self.storage_path}")
            except Exception as e:
                logger.error(f"Failed to load predictions: {e}")
                self.predictions = []
        else:
            logger.info(f"No existing predictions file at {self.storage_path}")

    def _save_predictions(self):
        """Save predictions to storage"""
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.storage_path, 'w') as f:
                data = [p.model_dump(mode='json') for p in self.predictions]
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {len(self.predictions)} predictions to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")

    def store_prediction(
        self,
        market_question: str,
        market_slug: str,
        predicted_probability: float,
        confidence_interval_low: float,
        confidence_interval_high: float,
        confidence_level: float = 0.80,
        market_url: Optional[str] = None,
        provider: Optional[str] = None,
        market_price_at_prediction: Optional[float] = None,
        workflow_id: Optional[str] = None,
        **kwargs
    ) -> PredictionRecord:
        """
        Store a new prediction

        Args:
            market_question: The market question
            market_slug: Market identifier
            predicted_probability: Our p_bayesian estimate
            confidence_interval_low: Lower bound of CI
            confidence_interval_high: Upper bound of CI
            confidence_level: Confidence level (e.g., 0.80 for 80%)
            market_url: Optional market URL
            provider: Optional provider name
            market_price_at_prediction: Optional market price
            workflow_id: Optional workflow ID
            **kwargs: Additional metadata

        Returns:
            PredictionRecord object
        """
        import uuid

        # Calculate edge if market price provided
        edge = None
        if market_price_at_prediction is not None:
            edge = predicted_probability - market_price_at_prediction

        prediction = PredictionRecord(
            prediction_id=str(uuid.uuid4()),
            market_question=market_question,
            market_slug=market_slug,
            market_url=market_url,
            provider=provider,
            predicted_probability=predicted_probability,
            confidence_interval_low=confidence_interval_low,
            confidence_interval_high=confidence_interval_high,
            confidence_level=confidence_level,
            market_price_at_prediction=market_price_at_prediction,
            edge=edge,
            workflow_id=workflow_id,
            **{k: v for k, v in kwargs.items() if k in PredictionRecord.model_fields}
        )

        self.predictions.append(prediction)
        self._save_predictions()

        logger.info(
            f"Stored prediction: {market_slug} = {predicted_probability:.2%} "
            f"[{confidence_interval_low:.2%} - {confidence_interval_high:.2%}]"
        )

        return prediction

    def update_outcome(
        self,
        prediction_id: str = None,
        market_slug: str = None,
        actual_outcome: bool = None,
        resolution_date: Optional[date] = None,
        resolution_source: Optional[str] = None
    ):
        """
        Update a prediction with actual outcome

        Args:
            prediction_id: Prediction ID (if known)
            market_slug: Market slug (alternative to prediction_id)
            actual_outcome: True if event happened, False if not
            resolution_date: Date when market resolved
            resolution_source: Source of resolution
        """
        # Find prediction
        prediction = None
        if prediction_id:
            prediction = next((p for p in self.predictions if p.prediction_id == prediction_id), None)
        elif market_slug:
            # Find most recent unresolved prediction for this market
            unresolved = [p for p in self.predictions if p.market_slug == market_slug and not p.resolved]
            prediction = unresolved[-1] if unresolved else None

        if not prediction:
            logger.warning(f"Prediction not found: prediction_id={prediction_id}, market_slug={market_slug}")
            return

        # Update prediction
        prediction.resolved = True
        prediction.actual_outcome = actual_outcome
        prediction.resolution_date = resolution_date or date.today()
        prediction.resolution_source = resolution_source

        self._save_predictions()

        logger.info(
            f"Updated outcome for {prediction.market_slug}: "
            f"predicted={prediction.predicted_probability:.2%}, actual={actual_outcome}"
        )

    def compute_metrics(self, min_resolved: int = 10) -> Optional[CalibrationMetrics]:
        """
        Compute calibration metrics from resolved predictions

        Args:
            min_resolved: Minimum number of resolved predictions required

        Returns:
            CalibrationMetrics or None if insufficient data
        """
        resolved = [p for p in self.predictions if p.resolved and p.actual_outcome is not None]

        if len(resolved) < min_resolved:
            logger.warning(
                f"Insufficient resolved predictions for calibration metrics: "
                f"{len(resolved)} < {min_resolved}"
            )
            return None

        total = len(self.predictions)
        unresolved = total - len(resolved)

        # Compute Brier score
        brier_score = sum(
            (p.predicted_probability - (1.0 if p.actual_outcome else 0.0)) ** 2
            for p in resolved
        ) / len(resolved)

        # Compute calibration buckets
        buckets = {
            "0-10%": [],
            "10-20%": [],
            "20-30%": [],
            "30-40%": [],
            "40-50%": [],
            "50-60%": [],
            "60-70%": [],
            "70-80%": [],
            "80-90%": [],
            "90-100%": []
        }

        for p in resolved:
            bucket_idx = int(p.predicted_probability * 10)
            bucket_idx = min(bucket_idx, 9)  # Cap at 90-100%
            bucket_labels = list(buckets.keys())
            bucket = bucket_labels[bucket_idx]
            buckets[bucket].append(1.0 if p.actual_outcome else 0.0)

        calibration_buckets = {}
        for bucket, outcomes in buckets.items():
            if outcomes:
                calibration_buckets[bucket] = {
                    "predicted_avg": (int(bucket.split('-')[0][:-1]) + int(bucket.split('-')[1][:-1])) / 2 / 100,
                    "actual_rate": sum(outcomes) / len(outcomes),
                    "count": len(outcomes)
                }

        # Compute edge accuracy
        positive_edges = [p for p in resolved if p.edge and p.edge > 0]
        negative_edges = [p for p in resolved if p.edge and p.edge < 0]

        positive_edge_win_rate = None
        if positive_edges:
            positive_edge_win_rate = sum(1 for p in positive_edges if p.actual_outcome) / len(positive_edges)

        negative_edge_avoid_rate = None
        if negative_edges:
            negative_edge_avoid_rate = sum(1 for p in negative_edges if not p.actual_outcome) / len(negative_edges)

        # Compute CI coverage
        ci_coverage = [
            p for p in resolved
            if p.confidence_interval_low <= (1.0 if p.actual_outcome else 0.0) <= p.confidence_interval_high
        ]
        ci_coverage_rate = len(ci_coverage) / len(resolved) if resolved else None

        metrics = CalibrationMetrics(
            total_predictions=total,
            resolved_predictions=len(resolved),
            unresolved_predictions=unresolved,
            brier_score=brier_score,
            calibration_buckets=calibration_buckets,
            positive_edge_win_rate=positive_edge_win_rate,
            negative_edge_avoid_rate=negative_edge_avoid_rate,
            ci_coverage_rate=ci_coverage_rate
        )

        logger.info(f"Computed calibration metrics: Brier={brier_score:.4f}, CI coverage={ci_coverage_rate:.2%}")

        return metrics

    def generate_report(self) -> str:
        """
        Generate a markdown calibration report

        Returns:
            Markdown string with calibration analysis
        """
        metrics = self.compute_metrics(min_resolved=1)  # Allow even 1 for report

        if not metrics:
            return "# Calibration Report\n\nInsufficient data for calibration analysis.\n"

        report = f"""# Calibration Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Predictions**: {metrics.total_predictions}
- **Resolved**: {metrics.resolved_predictions}
- **Unresolved**: {metrics.unresolved_predictions}
- **Brier Score**: {metrics.brier_score:.4f} (lower is better, 0 = perfect)

## Calibration by Probability Bucket

| Bucket | Predicted | Actual Rate | Count | Difference |
|--------|-----------|-------------|-------|------------|
"""

        for bucket, data in sorted(metrics.calibration_buckets.items()):
            predicted = data['predicted_avg']
            actual = data['actual_rate']
            count = data['count']
            diff = actual - predicted
            report += f"| {bucket} | {predicted:.1%} | {actual:.1%} | {count} | {diff:+.1%} |\n"

        report += f"""
## Edge Performance

"""

        if metrics.positive_edge_win_rate is not None:
            report += f"- **Positive Edge Win Rate**: {metrics.positive_edge_win_rate:.1%} (should be >50%)\n"
        else:
            report += "- **Positive Edge Win Rate**: N/A (no positive edges)\n"

        if metrics.negative_edge_avoid_rate is not None:
            report += f"- **Negative Edge Avoid Rate**: {metrics.negative_edge_avoid_rate:.1%} (should be >50%)\n"
        else:
            report += "- **Negative Edge Avoid Rate**: N/A (no negative edges)\n"

        report += f"""
## Confidence Interval Coverage

- **CI Coverage Rate**: {metrics.ci_coverage_rate:.1%} (target ~{self.predictions[0].confidence_level:.0%} for {int(self.predictions[0].confidence_level*100)}% CI)

## Interpretation

**Brier Score**: {self._interpret_brier(metrics.brier_score)}

**Calibration**: {self._interpret_calibration(metrics.calibration_buckets)}

**Edge Performance**: {self._interpret_edge(metrics.positive_edge_win_rate, metrics.negative_edge_avoid_rate)}

**CI Coverage**: {self._interpret_ci(metrics.ci_coverage_rate)}
"""

        return report

    def _interpret_brier(self, score: float) -> str:
        """Interpret Brier score"""
        if score < 0.05:
            return "Excellent - predictions are very well calibrated"
        elif score < 0.15:
            return "Good - predictions are reasonably calibrated"
        elif score < 0.25:
            return "Fair - some calibration issues"
        else:
            return "Poor - significant calibration problems"

    def _interpret_calibration(self, buckets: Dict) -> str:
        """Interpret calibration buckets"""
        if not buckets:
            return "Insufficient data"

        max_diff = max(abs(b['actual_rate'] - b['predicted_avg']) for b in buckets.values())

        if max_diff < 0.05:
            return "Excellent - actual rates closely match predictions across all buckets"
        elif max_diff < 0.10:
            return "Good - actual rates reasonably match predictions"
        elif max_diff < 0.20:
            return "Fair - some buckets show meaningful deviations"
        else:
            return "Poor - large deviations between predicted and actual rates"

    def _interpret_edge(self, pos_rate: Optional[float], neg_rate: Optional[float]) -> str:
        """Interpret edge performance"""
        if pos_rate is None and neg_rate is None:
            return "Insufficient data"

        results = []
        if pos_rate is not None:
            if pos_rate > 0.60:
                results.append("Strong positive edge detection")
            elif pos_rate > 0.50:
                results.append("Weak positive edge detection")
            else:
                results.append("Poor positive edge detection (losing money)")

        if neg_rate is not None:
            if neg_rate > 0.60:
                results.append("good negative edge avoidance")
            elif neg_rate > 0.50:
                results.append("weak negative edge avoidance")
            else:
                results.append("poor negative edge avoidance")

        return ", ".join(results)

    def _interpret_ci(self, coverage: Optional[float]) -> str:
        """Interpret CI coverage"""
        if coverage is None:
            return "Insufficient data"

        target = 0.80  # Assuming 80% CI
        if abs(coverage - target) < 0.05:
            return f"Excellent - coverage very close to target {target:.0%}"
        elif abs(coverage - target) < 0.10:
            return f"Good - coverage reasonably close to target"
        else:
            direction = "overconfident" if coverage < target else "underconfident"
            return f"Poor - {direction} intervals (coverage {coverage:.0%} vs target {target:.0%})"


# Convenience functions
def get_calibration_tracker(storage_path: str = "predictions.json") -> CalibrationTracker:
    """Get or create a calibration tracker"""
    return CalibrationTracker(storage_path=storage_path)
