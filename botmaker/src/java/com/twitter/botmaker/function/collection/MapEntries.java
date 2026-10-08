package com.twitter.botmaker.function.collection;

import java.util.Map;
import java.util.Set;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.tuples.Pair;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode1;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    name = {"MapEntries", "DictEntries"},
    argTypes = {"Map<A, B>"},
    arguments = {"dict"},
    returnType = "Set<Pair<A, B>>",
    description = "Returns a set of (key, value) pairs of a given dict",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "MapEntries(ToMap(123, \"asdf\", 456, \"jkl;\"))"
    }
)
public class MapEntries extends FunctionNode1<Runtime, Map> {

  private static final Type TPA = Type.newGenericType();
  private static final Type TPB = Type.newGenericType();
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.mapOf(TPA, TPB)),
      Type.setOf(Type.pairOf(TPA, TPB)));

  public MapEntries(String exprText, ImmutableList<ASTNode> children)
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
  protected Object apply(Context<Runtime> context, Map map) {
    ImmutableSet.Builder builder = ImmutableSet.builder();
    for (Map.Entry entry : (Set<Map.Entry>) map.entrySet()) {
      builder.add(Pair.of(entry.getKey(), entry.getValue()));
    }
    return builder.build();
  }
}
