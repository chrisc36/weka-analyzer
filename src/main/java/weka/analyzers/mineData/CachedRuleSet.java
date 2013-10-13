package weka.analyzers.mineData;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Abstract Class for rules that are composed of other rules.
 *
 * @param <T>  The type of Rule this RuleSet is a combination of
 */
public abstract class CachedRuleSet<T extends CachedRule> implements
        CachedRule, Iterable<T> {

    /** The modifier to deliminate rules with when printing this rule **/
    protected final String modifier;

    /** List of Rules this RuleSet is composed of */
    protected final List<T> rules;

    /** BitSet where set bits mark covered instances */
    protected BitSet covered;

    /**
     * Creates a RuleSet with no rules. Covers all instances.
     *
     * @param numInstances number of instances this RuleSet will apply to
     * @param modifier the String to use to deliminate Rules when printing
     */
    public CachedRuleSet(int numInstances, String modifier) {
        rules = new ArrayList<T>();
        this.modifier = modifier;
        covered = new BitSet(numInstances);
        covered.flip(0, numInstances);
    }

    /**
     * Adds a new Rules to this RuleSet.
     *
     * @param rule rule to add
     */
    public void add(T rule) {
        addBitSet(rule.covered());
        rules.add(rule);
    }

    /**
     * Adds a new Rule at the given index.
     *
     * @param index index to add the rule to
     * @param rule Rule to add
     */
    public void add(int index, T rule) {
        addBitSet(rule.covered());
        rules.add(index, rule);
    }

    /**
     * Reverse the order the Rules are stored in.
     */
    public void reverse() {
        Collections.reverse(rules);
    }

    /**
     * Gets the number of rules in this RuleSet.

     * @return the number of rules
     */
    public int size() {
        return rules.size();
    }

    /**
     * Return the Rule stored at a given index.
     *
     * @param index the index to use
     * @return the Rule
     */
    public T get(int index) {
        return rules.get(index);
    }

    /**
     * Get and Remove the Rule stored at a index.
     * Requires recalculating what instances this rules
     * covers from scratch.
     *
     * @param index to remove the rule the From
     * @return the Rule removed
     */
    public T remove(int index) {
        T r = rules.remove(index);
        recalculateCovered();
        return r;
    }

    /**
     * Swaps two rules in the list.
     *
     * @param index1 index of the first rule to swap
     * @param index2 index of the second rule to swap
     */
    public void swap(int index1, int index2) {
        T tmp = rules.get(index1);
        rules.set(index1, rules.get(index2));
        rules.set(index2, tmp);
    }

    /**
     * Recalculate what this rule covers from scratch.
     */
    private void recalculateCovered() {
        covered.clear();
        covered.flip(0, covered.size());
        for(CachedRule r : rules) {
            addBitSet(r.covered());
        }
    }

    /**
     * Get an Iterator over the Rules in this
     *
     * @return Rule Iterator of the rules in this
     */
    public Iterator<T> iterator() {
        return rules.iterator();
    }

    @Override
    public BitSet covered() {
        return covered;
    }

    @Override
    public String toString() {
        if(rules.size() == 0) {
            return "(empty)";
        }
        StringBuilder sb = new StringBuilder();
        if(rules.size() > 1) {
            sb.append("(");
        }
        sb.append(rules.get(0).toString());
        for(int i = 1; i < rules.size(); i++) {
            sb.append("\n\t" + modifier + " ");
            sb.append(rules.get(i).toString());
        }
        if(rules.size() > 1) {
            sb.append(")");
        }
        return sb.toString();
    }

    /**
     * Modifies covered so that it accounts for the addition of a new rule
     * assuming cover reflects what all the rules so far cover.
     *
     * @param other BitSet of what the new rule covers
     */
    protected abstract void addBitSet(BitSet other);
}
