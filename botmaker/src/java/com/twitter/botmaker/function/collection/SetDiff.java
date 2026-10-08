package com.twitter.botmaker.function.collection;

import java.util.Set;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode2;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    argTypes = {
        "Set<T>",
        "Set<T>"
    },
    arguments = {
        "A",
        "B"
    },
    returnType = "Set<T>",
    description = "The elements in set A but not in set B.",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "SetDiff(Set(5, 6, 7), Set(100, 12, 7))"
    }
)
public class SetDiff extends FunctionNode2<Runtime, Set, Set> {

  private static final Type TPA = Type.newGenericType();
  private static final Type TPS = Type.setOf(TPA);
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(TPS, TPS),
      TPS);

  public SetDiff(String exprText, ImmutableList<ASTNode> children)
      throws SemanticCheckFailure {
    super(exprText, children);
  }

  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  @Override
  @SuppressWarnings("unchecked")
  protected Object apply(Context<Runtime> context, Set set1, Set set2) {
    return Sets.difference(set1, set2);
  }
}
